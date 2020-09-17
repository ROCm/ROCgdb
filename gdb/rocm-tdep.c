/* Target-dependent code for ROCm.

   Copyright (C) 2019-2020 Free Software Foundation, Inc.
   Copyright (C) 2019-2020 Advanced Micro Devices, Inc. All rights reserved.

   This file is part of GDB.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

#include "defs.h"

#include "amdgcn-rocm-tdep.h"
#include "arch-utils.h"
#include "cli/cli-style.h"
#include "environ.h"
#include "event-loop.h"
#include "filenames.h"
#include "gdbcmd.h"
#include "gdbcore.h"
#include "gdbsupport/filestuff.h"
#include "gdbsupport/scoped_fd.h"
#include "gdbthread.h"
#include "hashtab.h"
#include "inf-loop.h"
#include "inferior.h"
#include "location.h"
#include "objfiles.h"
#include "observable.h"
#include "progspace-and-thread.h"
#include "regcache.h"
#include "rocm-tdep.h"
#include "solib.h"
#include "solist.h"
#include "symfile.h"

#include <dlfcn.h>
#include <list>
#include <map>
#include <set>
#include <signal.h>
#include <stdarg.h>
#include <unordered_map>

#include <amd-dbgapi.h>

/* Big enough to hold the size of the largest register in bytes.  */
#define AMDGCN_MAX_REGISTER_SIZE 256

#define DEFINE_OBSERVABLE(name) decltype (name) name (#name)

DEFINE_OBSERVABLE (amd_dbgapi_activated);
DEFINE_OBSERVABLE (amd_dbgapi_deactivated);
DEFINE_OBSERVABLE (amd_dbgapi_code_object_list_updated);

#undef DEFINE_OBSERVABLE

struct rocm_notify_shared_library_info
{
  std::string compare; /* Compare loaded library names with this string.  */
  struct so_list *solib;
};

/* ROCm-specific inferior data.  */

struct rocm_inferior_info
{
  /* True if the target is activated.  */
  bool activated{ false };

  /* The amd_dbgapi_process_id for this inferior.  */
  amd_dbgapi_process_id_t process_id{ AMD_DBGAPI_PROCESS_NONE };

  /* The amd_dbgapi_notifier_t for this inferior.  */
  amd_dbgapi_notifier_t notifier{ -1 };

  /* True if commit_resume should all-start the GPU queues.  */
  bool commit_resume_all_start;

  /* True if the inferior has exited.  */
  bool has_exited{ false };

  std::unordered_map<decltype (amd_dbgapi_breakpoint_id_t::handle),
                     struct breakpoint *>
      breakpoint_map;

  std::map<CORE_ADDR, std::pair<CORE_ADDR, amd_dbgapi_watchpoint_id_t>>
      watchpoint_map;

  /* List of pending events the rocm target retrieved from the dbgapi.  */
  std::list<std::pair<amd_dbgapi_wave_id_t, amd_dbgapi_wave_stop_reason_t>>
      wave_stop_events;

  /* Map of rocm_notify_shared_library_info's for libraries that have been
     registered to receive notifications when loading/unloading.  */
  std::unordered_map<decltype (amd_dbgapi_shared_library_id_t::handle),
                     struct rocm_notify_shared_library_info>
      notify_solib_map;
};

static amd_dbgapi_event_id_t
rocm_process_event_queue (amd_dbgapi_event_kind_t until_event_kind
                          = AMD_DBGAPI_EVENT_KIND_NONE);

/* Return the inferior's rocm_inferior_info struct.  */
static struct rocm_inferior_info *
get_rocm_inferior_info (struct inferior *inferior = nullptr);

static const target_info rocm_ops_info
    = { "rocm", N_ ("ROCm GPU debugging support"),
        N_ ("ROCm GPU debugging support") };

static amd_dbgapi_log_level_t get_debug_amdgpu_log_level ();

struct rocm_target_ops final : public target_ops
{
  const target_info &
  info () const override
  {
    return rocm_ops_info;
  }
  strata
  stratum () const override
  {
    return arch_stratum;
  }

  void close () override;
  void mourn_inferior () override;
  void detach (inferior *inf, int from_tty) override;

  void async (int enable) override;

  ptid_t wait (ptid_t, struct target_waitstatus *, int) override;
  void resume (ptid_t, int, enum gdb_signal) override;
  void commit_resume () override;
  void stop (ptid_t ptid) override;

  void fetch_registers (struct regcache *, int) override;
  void store_registers (struct regcache *, int) override;

  void update_thread_list () override;

  struct gdbarch *thread_architecture (ptid_t) override;

  std::string pid_to_str (ptid_t ptid) override;

  const char *thread_name (thread_info *tp) override;

  const char *extra_thread_info (thread_info *tp) override;

  bool thread_alive (ptid_t ptid) override;

  enum target_xfer_status xfer_partial (enum target_object object,
                                        const char *annex, gdb_byte *readbuf,
                                        const gdb_byte *writebuf,
                                        ULONGEST offset, ULONGEST len,
                                        ULONGEST *xfered_len) override;

  int insert_watchpoint (CORE_ADDR addr, int len, enum target_hw_bp_type type,
                         struct expression *cond) override;
  int remove_watchpoint (CORE_ADDR addr, int len, enum target_hw_bp_type type,
                         struct expression *cond) override;
  bool stopped_by_watchpoint () override;
  bool stopped_data_address (CORE_ADDR *addr_p) override;

  bool stopped_by_sw_breakpoint () override;
  bool stopped_by_hw_breakpoint () override;
};

/* ROCm's target vector.  */
static struct rocm_target_ops rocm_ops;

/* ROCm breakpoint ops.  */
static struct breakpoint_ops rocm_breakpoint_ops;

/* Per-inferior data key.  */
static const struct inferior_key<rocm_inferior_info> rocm_inferior_data;

/* The read/write ends of the pipe registered as waitable file in the
   event loop.  */
static int rocm_event_pipe[2] = { -1, -1 };

/* Flush the event pipe.  */

static void
async_file_flush (void)
{
  int ret;
  char buf;

  do
    {
      ret = read (rocm_event_pipe[0], &buf, 1);
    }
  while (ret >= 0 || (ret == -1 && errno == EINTR));
}

/* Put something (anything, doesn't matter what, or how much) in event
   pipe, so that the select/poll in the event-loop realizes we have
   something to process.  */

static void
async_file_mark (void)
{
  int ret;

  /* It doesn't really matter what the pipe contains, as long we end
     up with something in it.  Might as well flush the previous
     left-overs.  */
  async_file_flush ();

  do
    {
      ret = write (rocm_event_pipe[1], "+", 1);
    }
  while (ret == -1 && errno == EINTR);

  /* Ignore EAGAIN.  If the pipe is full, the event loop will already
     be awakened anyway.  */
}

/* Fetch the rocm_inferior_info data for the given inferior.  */

static struct rocm_inferior_info *
get_rocm_inferior_info (struct inferior *inferior)
{
  if (!inferior)
    inferior = current_inferior ();

  struct rocm_inferior_info *info = rocm_inferior_data.get (inferior);

  if (!info)
    info = rocm_inferior_data.emplace (inferior);

  return info;
}

/* Fetch the amd_dbgapi_process_id for the given inferior.  */

amd_dbgapi_process_id_t
get_amd_dbgapi_process_id (struct inferior *inferior)
{
  return get_rocm_inferior_info (inferior)->process_id;
}

static void
rocm_breakpoint_re_set (struct breakpoint *b)
{
}

static void
rocm_breakpoint_check_status (struct bpstats *bs)
{
  struct rocm_inferior_info *info = get_rocm_inferior_info ();
  amd_dbgapi_process_id_t process_id = info->process_id;
  amd_dbgapi_status_t status;

  bs->stop = 0;
  bs->print_it = print_it_noop;

  /* Find the address the breakpoint is set at.  */
  auto it = std::find_if (
      info->breakpoint_map.begin (), info->breakpoint_map.end (),
      [=] (const decltype (info->breakpoint_map)::value_type &value) {
        return value.second == bs->breakpoint_at;
      });

  if (it == info->breakpoint_map.end ())
    error (_ ("Could not find breakpoint_id for breakpoint at %#lx"),
           bs->bp_location_at->address);

  amd_dbgapi_breakpoint_id_t breakpoint_id{ it->first };
  amd_dbgapi_breakpoint_action_t action;

  status = amd_dbgapi_report_breakpoint_hit (
      process_id, breakpoint_id,
      reinterpret_cast<amd_dbgapi_client_thread_id_t> (inferior_thread ()),
      &action);

  if (status != AMD_DBGAPI_STATUS_SUCCESS)
    error (_ ("amd_dbgapi_report_breakpoint_hit failed: breakpoint_%ld "
              "at %#lx (rc=%d)"),
           breakpoint_id.handle, bs->bp_location_at->address, status);

  if (action == AMD_DBGAPI_BREAKPOINT_ACTION_RESUME)
    return;

  /* If the action is AMD_DBGAPI_BREAKPOINT_ACTION_HALT, we need to wait until
     a breakpoint resume event for this breakpoint_id is seen.  */

  amd_dbgapi_event_id_t resume_event_id
      = rocm_process_event_queue (AMD_DBGAPI_EVENT_KIND_BREAKPOINT_RESUME);

  /* We should always get a breakpoint_resume event after processing all
     events generated by reporting the breakpoint was hit.  */
  gdb_assert (resume_event_id.handle != AMD_DBGAPI_EVENT_NONE.handle);

  amd_dbgapi_breakpoint_id_t resume_breakpoint_id;
  status = amd_dbgapi_event_get_info (
      process_id, resume_event_id, AMD_DBGAPI_EVENT_INFO_BREAKPOINT,
      sizeof (resume_breakpoint_id), &resume_breakpoint_id);

  if (status != AMD_DBGAPI_STATUS_SUCCESS)
    error (_ ("amd_dbgapi_event_get_info failed (rc=%d)"), status);

  /* The debugger API guarantees that [breakpoint_hit...resume_breakpoint]
     sequences cannot interleave, so this breakpoint resume event must be
     for our breakpoint_id.  */
  if (resume_breakpoint_id.handle != breakpoint_id.handle)
    error (_ ("breakpoint resume event is not for this breakpoint. "
              "Expected breakpoint_%ld, got breakpoint_%ld"),
           breakpoint_id.handle, resume_breakpoint_id.handle);

  amd_dbgapi_event_processed (process_id, resume_event_id);
}

static void
rocm_target_dbgapi_activated ()
{
  /* FIXME: only push on the first activation.  */
  /* Engage the ROCm target_ops and so_ops.  */
  push_target (&rocm_ops);
}

static void
rocm_target_dbgapi_deactivated ()
{
  /* FIXME: only unpush on the last activation.  */
  /* Disengage the ROCm target_ops.  */
  unpush_target (&rocm_ops);
}

bool
rocm_target_ops::thread_alive (ptid_t ptid)
{
  if (!ptid_is_gpu (ptid))
    return beneath ()->thread_alive (ptid);

  /* Check that the wave_id is valid.  */

  inferior *inf = find_inferior_ptid (ptid);
  if (!inf)
    return false;

  amd_dbgapi_wave_state_t state;
  return amd_dbgapi_wave_get_info (
             get_amd_dbgapi_process_id (inf), get_amd_dbgapi_wave_id (ptid),
             AMD_DBGAPI_WAVE_INFO_STATE, sizeof (state), &state)
         == AMD_DBGAPI_STATUS_SUCCESS;
}

const char *
rocm_target_ops::thread_name (thread_info *tp)
{
  if (!ptid_is_gpu (tp->ptid))
    return beneath ()->thread_name (tp);

  /* Return the process's comm value—that is, the command name associated with
     the process.  */

  char comm_path[128];
  xsnprintf (comm_path, sizeof (comm_path), "/proc/%ld/comm",
             (long)tp->ptid.pid ());

  gdb_file_up comm_file = gdb_fopen_cloexec (comm_path, "r");
  if (!comm_file)
    return nullptr;

#if !defined(TASK_COMM_LEN)
#define TASK_COMM_LEN 16 /* As defined in the kernel's sched.h.  */
#endif

  static char comm_buf[TASK_COMM_LEN];
  const char *comm_value;

  comm_value = fgets (comm_buf, sizeof (comm_buf), comm_file.get ());
  comm_buf[sizeof (comm_buf) - 1] = '\0';

  /* Make sure there is no newline at the end.  */
  if (comm_value)
    {
      for (int i = 0; i < sizeof (comm_buf); i++)
        if (comm_buf[i] == '\n')
          {
            comm_buf[i] = '\0';
            break;
          }
    }

  return comm_value;
}

std::string
rocm_target_ops::pid_to_str (ptid_t ptid)
{
  if (!ptid_is_gpu (ptid))
    {
      return beneath ()->pid_to_str (ptid);
    }

  amd_dbgapi_process_id_t process_id = get_amd_dbgapi_process_id ();
  amd_dbgapi_wave_id_t wave_id = get_amd_dbgapi_wave_id (ptid);
  amd_dbgapi_dispatch_id_t dispatch_id;
  uint32_t group_ids[3], wave_in_group;

  if (amd_dbgapi_wave_get_info (process_id, wave_id,
                                AMD_DBGAPI_WAVE_INFO_DISPATCH,
                                sizeof (dispatch_id), &dispatch_id)
          != AMD_DBGAPI_STATUS_SUCCESS
      || amd_dbgapi_wave_get_info (process_id, wave_id,
                                   AMD_DBGAPI_WAVE_INFO_WORK_GROUP_COORD,
                                   sizeof (group_ids), &group_ids)
             != AMD_DBGAPI_STATUS_SUCCESS
      || amd_dbgapi_wave_get_info (
             process_id, wave_id,
             AMD_DBGAPI_WAVE_INFO_WAVE_NUMBER_IN_WORK_GROUP,
             sizeof (wave_in_group), &wave_in_group)
             != AMD_DBGAPI_STATUS_SUCCESS)
    return std::string ("AMDGPU Thread");
  else
    return string_printf ("AMDGPU Thread %ld.%ld (%d,%d,%d)/%d",
                          dispatch_id.handle, wave_id.handle, group_ids[2],
                          group_ids[1], group_ids[0], wave_in_group);
}

const char *
rocm_target_ops::extra_thread_info (thread_info *tp)
{
  if (!ptid_is_gpu (tp->ptid))
    beneath ()->extra_thread_info (tp);

  return NULL;
}

enum target_xfer_status
rocm_target_ops::xfer_partial (enum target_object object, const char *annex,
                               gdb_byte *readbuf, const gdb_byte *writebuf,
                               ULONGEST offset, ULONGEST requested_len,
                               ULONGEST *xfered_len)
{
  gdb::optional<scoped_restore_current_thread> maybe_restore_thread;

  if (ptid_is_gpu (inferior_ptid))
    {
      gdb_assert (requested_len && xfered_len && "checking invariants");

      if (object != TARGET_OBJECT_MEMORY)
        return TARGET_XFER_E_IO;

      /* FIXME: We current have no way to specify the address space, so it is
         encoded in the "unused" bits of a canonical address.  */
      uint64_t dwarf_address_space
          = (offset & ROCM_ASPACE_MASK) >> ROCM_ASPACE_BIT_OFFSET;

      amd_dbgapi_segment_address_t segment_address
          = offset & ~ROCM_ASPACE_MASK;

      /* FIXME: Default to the generic address space to allow the examine
         command to work with flat addresses wihout special syntax.  */
      if (!dwarf_address_space)
        dwarf_address_space = /*DW_ASPACE_AMDGPU_generic*/ 1;

      amd_dbgapi_process_id_t process_id = get_amd_dbgapi_process_id ();
      amd_dbgapi_wave_id_t wave_id = get_amd_dbgapi_wave_id (inferior_ptid);

      amd_dbgapi_architecture_id_t architecture_id;
      amd_dbgapi_address_space_id_t address_space_id;

      if (amd_dbgapi_wave_get_info (process_id, wave_id,
                                    AMD_DBGAPI_WAVE_INFO_ARCHITECTURE,
                                    sizeof (architecture_id), &architecture_id)
              != AMD_DBGAPI_STATUS_SUCCESS
          || amd_dbgapi_dwarf_address_space_to_address_space (
                 architecture_id, dwarf_address_space, &address_space_id)
                 != AMD_DBGAPI_STATUS_SUCCESS)
        return TARGET_XFER_EOF;

      size_t len = requested_len;
      amd_dbgapi_status_t status;
      if (readbuf)
        status
            = amd_dbgapi_read_memory (process_id, wave_id, 0, address_space_id,
                                      segment_address, &len, readbuf);
      else
        status = amd_dbgapi_write_memory (process_id, wave_id, 0,
                                          address_space_id, segment_address,
                                          &len, writebuf);

      if (status != AMD_DBGAPI_STATUS_SUCCESS)
        return TARGET_XFER_EOF;

      *xfered_len = len;
      return TARGET_XFER_OK;
    }
  else
    return beneath ()->xfer_partial (object, annex, readbuf, writebuf, offset,
                                     requested_len, xfered_len);
}

int
rocm_target_ops::insert_watchpoint (CORE_ADDR addr, int len,
                                    enum target_hw_bp_type type,
                                    struct expression *cond)
{
  struct rocm_inferior_info *info = get_rocm_inferior_info ();

  if (type != hw_write /* for now, we only allow write watchpoints.  */
      || beneath ()->insert_watchpoint (addr, len, type, cond))
    return 1;

  amd_dbgapi_watchpoint_id_t watch_id;
  amd_dbgapi_global_address_t adjusted_address;
  amd_dbgapi_size_t adjusted_size;

  if (amd_dbgapi_set_watchpoint (info->process_id, addr, len,
                                 AMD_DBGAPI_WATCHPOINT_KIND_STORE_AND_RMW,
                                 &watch_id, &adjusted_address, &adjusted_size)
          != AMD_DBGAPI_STATUS_SUCCESS
      /* FIXME: A reduced range watchpoint may have been inserted, which would
        require additional watchpoints to be inserted to cover the requested
        range.  */
      || adjusted_address > addr
      || (adjusted_address + adjusted_size) < (addr + len))
    {
      /* We failed to insert the GPU watchpoint, so remove the CPU watchpoint
         before returning an error.  */
      beneath ()->remove_watchpoint (addr, len, type, cond);
      return 1;
    }

  if (!info->watchpoint_map
           .emplace (addr, std::make_pair (addr + len, watch_id))
           .second)
    {
      amd_dbgapi_remove_watchpoint (info->process_id, watch_id);
      beneath ()->remove_watchpoint (addr, len, type, cond);
      return 1;
    }

  return 0;
}

int
rocm_target_ops::remove_watchpoint (CORE_ADDR addr, int len,
                                    enum target_hw_bp_type type,
                                    struct expression *cond)
{
  struct rocm_inferior_info *info = get_rocm_inferior_info ();
  gdb_assert (type == hw_write);

  if (beneath ()->remove_watchpoint (addr, len, type, cond))
    return 1;

  /* Find the watch_id for the addr..addr+len range.  */
  auto it = info->watchpoint_map.upper_bound (addr);
  if (it == info->watchpoint_map.begin ())
    return 1;

  std::advance (it, -1);
  if (addr >= it->first && (addr + len) <= it->second.first)
    {
      amd_dbgapi_watchpoint_id_t watch_id = it->second.second;
      info->watchpoint_map.erase (it);
      return amd_dbgapi_remove_watchpoint (info->process_id, watch_id)
             != AMD_DBGAPI_STATUS_SUCCESS;
    }

  return 1;
}

bool
rocm_target_ops::stopped_by_watchpoint ()
{
  if (!ptid_is_gpu (inferior_ptid))
    return beneath ()->stopped_by_watchpoint ();

  amd_dbgapi_watchpoint_list_t watchpoints;
  if (amd_dbgapi_wave_get_info (
          get_amd_dbgapi_process_id (), get_amd_dbgapi_wave_id (inferior_ptid),
          AMD_DBGAPI_WAVE_INFO_WATCHPOINTS, sizeof (watchpoints), &watchpoints)
      != AMD_DBGAPI_STATUS_SUCCESS)
    return false;

  free (watchpoints.watchpoint_ids);
  return watchpoints.count != 0;
}

bool
rocm_target_ops::stopped_data_address (CORE_ADDR *addr_p)
{
  struct rocm_inferior_info *info = get_rocm_inferior_info ();

  if (!ptid_is_gpu (inferior_ptid))
    return beneath ()->stopped_data_address (addr_p);

  amd_dbgapi_watchpoint_list_t watchpoints;
  if (amd_dbgapi_wave_get_info (
          get_amd_dbgapi_process_id (), get_amd_dbgapi_wave_id (inferior_ptid),
          AMD_DBGAPI_WAVE_INFO_WATCHPOINTS, sizeof (watchpoints), &watchpoints)
      != AMD_DBGAPI_STATUS_SUCCESS)
    return false;

  /* Compute the intersection between the triggered watchpoint ranges.  */
  CORE_ADDR start = std::numeric_limits<CORE_ADDR>::min ();
  CORE_ADDR finish = std::numeric_limits<CORE_ADDR>::max ();
  for (size_t i = 0; i < watchpoints.count; ++i)
    {
      amd_dbgapi_watchpoint_id_t watchpoint = watchpoints.watchpoint_ids[i];
      auto it = std::find_if (
          info->watchpoint_map.begin (), info->watchpoint_map.end (),
          [watchpoint] (
              const decltype (info->watchpoint_map)::value_type &value) {
            return value.second.second.handle == watchpoint.handle;
          });
      if (it != info->watchpoint_map.end ())
        {
          start = std::max (start, it->first);
          finish = std::min (finish, it->second.first);
        }
    }
  free (watchpoints.watchpoint_ids);

  /* infrun does not seem to care about the exact address, anything within
     the watched address range is good enough to identify the watchpoint.  */
  *addr_p = start;
  return start < finish;
}

void
rocm_target_ops::resume (ptid_t ptid, int step, enum gdb_signal signo)
{
  struct rocm_inferior_info *info = get_rocm_inferior_info ();

  if (debug_infrun)
    fprintf_unfiltered (
        gdb_stdlog,
        "\e[1;34minfrun: rocm_target_ops::resume ([%d,%ld,%ld])\e[0m\n",
        ptid.pid (), ptid.lwp (), ptid.tid ());

  /* Check if the thread focus is on the GPU device.  */
  if (ptid == minus_one_ptid || !ptid_is_gpu (ptid))
    {
      beneath ()->resume (ptid, step, signo);
      if (ptid != minus_one_ptid)
        return;
    }

  /* A specific PTID means `step only this process id'.  */
  bool resume_one = ptid != minus_one_ptid && !ptid.is_pid ();
  gdb_assert (resume_one || !step);

  if (!resume_one)
    error (_ ("internal error - unimplemented "));

  amd_dbgapi_process_set_progress (info->process_id,
                                   AMD_DBGAPI_PROGRESS_NO_FORWARD);

  amd_dbgapi_status_t status = amd_dbgapi_wave_resume (
      info->process_id, get_amd_dbgapi_wave_id (ptid),
      step ? AMD_DBGAPI_RESUME_MODE_SINGLE_STEP
           : AMD_DBGAPI_RESUME_MODE_NORMAL);
  if (status != AMD_DBGAPI_STATUS_SUCCESS)
    warning (_ ("Could not resume %s (rc=%d)"),
             target_pid_to_str (ptid).c_str (), status);

  info->commit_resume_all_start = true;
}

void
rocm_target_ops::commit_resume ()
{
  struct rocm_inferior_info *info = get_rocm_inferior_info ();

  if (debug_infrun)
    fprintf_unfiltered (
        gdb_stdlog,
        "\e[1;34minfrun: rocm_target_ops::commit_resume ()\e[0m\n");

  beneath ()->commit_resume ();

  if (info->commit_resume_all_start)
    {
      amd_dbgapi_process_set_progress (info->process_id,
                                       AMD_DBGAPI_PROGRESS_NORMAL);
      info->commit_resume_all_start = false;
    }

  if (target_can_async_p ())
    target_async (1);
}

static void
rocm_target_stop_one_wave (ptid_t ptid)
{
  amd_dbgapi_status_t status;

  status = amd_dbgapi_wave_stop (get_amd_dbgapi_process_id (),
                                 get_amd_dbgapi_wave_id (ptid));

  if (status == AMD_DBGAPI_STATUS_ERROR_INVALID_WAVE_ID)
    {
      /* the wave must have exited, set the thread status to reflect that.  */
      auto *tp = find_thread_ptid (ptid);
      gdb_assert (tp);

      tp->state = THREAD_EXITED;
    }
  else if (status != AMD_DBGAPI_STATUS_SUCCESS)
    warning (_ ("Could not stop %s (rc=%d)"),
             target_pid_to_str (ptid).c_str (), status);
}

void
rocm_target_ops::stop (ptid_t ptid)
{
  if (debug_infrun)
    fprintf_unfiltered (
        gdb_stdlog,
        "\e[1;34minfrun: rocm_target_ops::stop ([%d,%ld,%ld])\e[0m\n",
        ptid.pid (), ptid.lwp (), ptid.tid ());

  if (ptid == minus_one_ptid || !ptid_is_gpu (ptid))
    {
      beneath ()->stop (ptid);
      if (ptid != minus_one_ptid)
        return;
    }

  if (ptid == minus_one_ptid)
    error (_ ("internal error - unimplemented "));

  rocm_target_stop_one_wave (ptid);
}

static void
handle_target_event (int error, gdb_client_data client_data)
{
  struct rocm_inferior_info *info = get_rocm_inferior_info ();
  amd_dbgapi_process_id_t process_id = info->process_id;

  amd_dbgapi_process_set_progress (process_id, AMD_DBGAPI_PROGRESS_NO_FORWARD);

  /* Flush the async file first.  */
  if (target_is_async_p ())
    async_file_flush ();

  rocm_process_event_queue ();

  /* In all-stop mode, unless the event queue is empty (spurious wake-up),
     we can keep the process in progress_no_forward mode.  The infrun loop
     will enable forward progress when a thread is resumed.  */
  if (non_stop || info->wave_stop_events.empty ())
    amd_dbgapi_process_set_progress (process_id, AMD_DBGAPI_PROGRESS_NORMAL);

  if (!info->wave_stop_events.empty ())
    inferior_event_handler (INF_REG_EVENT, nullptr);
}

void
rocm_target_ops::async (int enable)
{
  beneath ()->async (enable);

  if (enable)
    {
      if (rocm_event_pipe[0] != -1)
        return;

      if (gdb_pipe_cloexec (rocm_event_pipe) == -1)
        internal_error (__FILE__, __LINE__, "creating event pipe failed.");

      ::fcntl (rocm_event_pipe[0], F_SETFL, O_NONBLOCK);
      ::fcntl (rocm_event_pipe[1], F_SETFL, O_NONBLOCK);

      add_file_handler (rocm_event_pipe[0], handle_target_event, nullptr);

      /* There may be pending events to handle.  Tell the event loop
         to poll them.  */
      async_file_mark ();
    }
  else
    {
      if (rocm_event_pipe[0] == -1)
        return;

      delete_file_handler (rocm_event_pipe[0]);

      ::close (rocm_event_pipe[0]);
      ::close (rocm_event_pipe[1]);
      rocm_event_pipe[0] = -1;
      rocm_event_pipe[1] = -1;
    }
}

static void
rocm_process_one_event (amd_dbgapi_event_id_t event_id,
                        amd_dbgapi_event_kind_t event_kind)
{
  struct rocm_inferior_info *info = get_rocm_inferior_info ();
  amd_dbgapi_process_id_t process_id = info->process_id;
  amd_dbgapi_status_t status;

  switch (event_kind)
    {
    case AMD_DBGAPI_EVENT_KIND_WAVE_STOP:
      {
        amd_dbgapi_wave_id_t wave_id;
        if ((status = amd_dbgapi_event_get_info (process_id, event_id,
                                                 AMD_DBGAPI_EVENT_INFO_WAVE,
                                                 sizeof (wave_id), &wave_id))
            != AMD_DBGAPI_STATUS_SUCCESS)
          error (_ ("event_get_info for event_%ld failed (rc=%d)"),
                 event_id.handle, status);

        amd_dbgapi_wave_stop_reason_t stop_reason;
        status = amd_dbgapi_wave_get_info (process_id, wave_id,
                                           AMD_DBGAPI_WAVE_INFO_STOP_REASON,
                                           sizeof (stop_reason), &stop_reason);

        if (status != AMD_DBGAPI_STATUS_SUCCESS
            && status != AMD_DBGAPI_STATUS_ERROR_INVALID_WAVE_ID)
          error (_ ("wave_get_info for wave_%ld failed (rc=%d)"),
                 wave_id.handle, status);

        /* The wave may have exited, or the queue went into an error
           state.  In such cases, we will see another wave command
           terminated event, and handle the wave termination then.  */

        if (status == AMD_DBGAPI_STATUS_SUCCESS)
          info->wave_stop_events.emplace_back (
              std::make_pair (wave_id, stop_reason));
      }
      break;

    case AMD_DBGAPI_EVENT_KIND_CODE_OBJECT_LIST_UPDATED:
      amd_dbgapi_code_object_list_updated.notify ();
      break;

    case AMD_DBGAPI_EVENT_KIND_BREAKPOINT_RESUME:
      /* Breakpoint resume events should be handled by the breakpoint
         action, and this code should not reach this.  */
      gdb_assert_not_reached (_ ("unhandled event kind"));
      break;

    case AMD_DBGAPI_EVENT_KIND_RUNTIME:
      {
        amd_dbgapi_runtime_state_t runtime_state;

        if ((status = amd_dbgapi_event_get_info (
                 process_id, event_id, AMD_DBGAPI_EVENT_INFO_RUNTIME_STATE,
                 sizeof (runtime_state), &runtime_state))
            != AMD_DBGAPI_STATUS_SUCCESS)
          error (_ ("event_get_info for event_%ld failed (rc=%d)"),
                 event_id.handle, status);

        switch (runtime_state)
          {
          case AMD_DBGAPI_RUNTIME_STATE_LOADED_SUCCESS:
            amd_dbgapi_activated.notify ();
            info->activated = true;
            break;

          case AMD_DBGAPI_RUNTIME_STATE_UNLOADED:
            if (info->activated)
              amd_dbgapi_deactivated.notify ();
            break;

          case AMD_DBGAPI_RUNTIME_STATE_LOADED_ERROR_RESTRICTION:
            error (_ ("ROCgdb: unable to enable GPU debugging "
                      "due to a restriction error"));
            break;
          }
      }
      break;

    default:
      error (_ ("event kind (%d) not supported"), event_kind);
    }

  amd_dbgapi_event_processed (process_id, event_id);
}

/* Drain the amd_dbgapi event queue until an event of the given type is seen.
   If no particular event kind is specified (AMD_DBGAPI_EVENT_KIND_NONE), the
   event queue is completely drained. Wave stop events that are not returned
   are re-queued into the current's process pending wave events. */
static amd_dbgapi_event_id_t
rocm_process_event_queue (amd_dbgapi_event_kind_t until_event_kind)
{
  struct rocm_inferior_info *info = get_rocm_inferior_info ();

  while (true)
    {
      amd_dbgapi_event_id_t event_id;
      amd_dbgapi_event_kind_t event_kind;

      amd_dbgapi_status_t status = amd_dbgapi_next_pending_event (
          info->process_id, &event_id, &event_kind);

      if (status != AMD_DBGAPI_STATUS_SUCCESS)
        error (_ ("next_pending_event failed (rc=%d)"), status);

      if (event_id.handle == AMD_DBGAPI_EVENT_NONE.handle
          || event_kind == until_event_kind)
        return event_id;

      rocm_process_one_event (event_id, event_kind);
    }
}

ptid_t
rocm_target_ops::wait (ptid_t ptid, struct target_waitstatus *ws,
                       int target_options)
{
  if (debug_infrun)
    fprintf_unfiltered (gdb_stdlog,
                        "\e[1;34minfrun: rocm_target_ops::wait\e[0m\n");

  if (!ptid_is_gpu (ptid))
    {
      ptid_t event_ptid = beneath ()->wait (ptid, ws, target_options);
      if (event_ptid != minus_one_ptid)
        return event_ptid;
    }

  struct rocm_inferior_info *info = get_rocm_inferior_info ();
  amd_dbgapi_process_id_t process_id = info->process_id;

  /* Drain all the events from the amd_dbgapi, and preserve the ordering.  */
  if (info->wave_stop_events.empty ())
    {
      amd_dbgapi_process_set_progress (process_id,
                                       AMD_DBGAPI_PROGRESS_NO_FORWARD);

      /* Flush the async file first.  */
      if (target_is_async_p ())
        async_file_flush ();

      rocm_process_event_queue ();

      /* In all-stop mode, unless the event queue is empty (spurious wake-up),
         we can keep the process in progress_no_forward mode.  The infrun loop
         will enable forward progress when a thread is resumed.  */
      if (non_stop || info->wave_stop_events.empty ())
        amd_dbgapi_process_set_progress (process_id,
                                         AMD_DBGAPI_PROGRESS_NORMAL);
    }

  if (info->wave_stop_events.empty ())
    return minus_one_ptid;

  amd_dbgapi_wave_id_t event_wave_id;
  amd_dbgapi_wave_stop_reason_t stop_reason;

  std::tie (event_wave_id, stop_reason) = info->wave_stop_events.front ();
  info->wave_stop_events.pop_front ();

  ptid_t event_ptid (current_inferior ()->pid, 1, event_wave_id.handle);

  if (!find_thread_ptid (event_ptid))
    {
      add_thread_silent (event_ptid);
      set_running (event_ptid, 1);
      set_executing (event_ptid, 1);
    }

  /* Since we are manipulating the register cache for the event thread,
     make sure it is the current thread.  */
  switch_to_thread (event_ptid);

  /* By caching the PC now, we avoid having to suspend/resume the queue
     later when we need to access it.  */
  amd_dbgapi_global_address_t stop_pc;
  if (amd_dbgapi_wave_get_info (process_id, event_wave_id,
                                AMD_DBGAPI_WAVE_INFO_PC, sizeof (stop_pc),
                                &stop_pc)
      == AMD_DBGAPI_STATUS_SUCCESS)
    {
      struct regcache *regcache = get_thread_regcache (event_ptid);
      regcache->raw_supply (gdbarch_pc_regnum (regcache->arch ()), &stop_pc);
    }
  ws->kind = TARGET_WAITKIND_STOPPED;

  if (stop_reason & AMD_DBGAPI_WAVE_STOP_REASON_MEMORY_VIOLATION)
    ws->value.sig = GDB_SIGNAL_SEGV;
  else if (stop_reason
           & (AMD_DBGAPI_WAVE_STOP_REASON_FP_INPUT_DENORMAL
              | AMD_DBGAPI_WAVE_STOP_REASON_FP_DIVIDE_BY_0
              | AMD_DBGAPI_WAVE_STOP_REASON_FP_OVERFLOW
              | AMD_DBGAPI_WAVE_STOP_REASON_FP_UNDERFLOW
              | AMD_DBGAPI_WAVE_STOP_REASON_FP_INEXACT
              | AMD_DBGAPI_WAVE_STOP_REASON_FP_INVALID_OPERATION
              | AMD_DBGAPI_WAVE_STOP_REASON_INT_DIVIDE_BY_0))
    ws->value.sig = GDB_SIGNAL_FPE;
  else if (stop_reason
           & (AMD_DBGAPI_WAVE_STOP_REASON_BREAKPOINT
              | AMD_DBGAPI_WAVE_STOP_REASON_WATCHPOINT
              | AMD_DBGAPI_WAVE_STOP_REASON_SINGLE_STEP
              | AMD_DBGAPI_WAVE_STOP_REASON_DEBUG_TRAP
              | AMD_DBGAPI_WAVE_STOP_REASON_ASSERT_TRAP
              | AMD_DBGAPI_WAVE_STOP_REASON_TRAP))
    ws->value.sig = GDB_SIGNAL_TRAP;
  else
    ws->value.sig = GDB_SIGNAL_0;

  /* If there are more events in the list, mark the async file so that
     rocm_target_ops::wait gets called again.  */
  if (target_is_async_p () && !info->wave_stop_events.empty ())
    async_file_mark ();

  return event_ptid;
}

bool
rocm_target_ops::stopped_by_sw_breakpoint ()
{
  if (!ptid_is_gpu (inferior_ptid))
    return beneath ()->stopped_by_sw_breakpoint ();

  amd_dbgapi_process_id_t process_id = get_amd_dbgapi_process_id ();
  amd_dbgapi_wave_id_t wave_id = get_amd_dbgapi_wave_id (inferior_ptid);

  amd_dbgapi_wave_stop_reason_t stop_reason;
  return (amd_dbgapi_wave_get_info (process_id, wave_id,
                                    AMD_DBGAPI_WAVE_INFO_STOP_REASON,
                                    sizeof (stop_reason), &stop_reason)
          == AMD_DBGAPI_STATUS_SUCCESS)
         && (stop_reason & AMD_DBGAPI_WAVE_STOP_REASON_BREAKPOINT) != 0;
}

bool
rocm_target_ops::stopped_by_hw_breakpoint ()
{
  /* The rocm target does not support hw breakpoints.  */
  return !ptid_is_gpu (inferior_ptid)
         && beneath ()->stopped_by_hw_breakpoint ();
}

void
rocm_target_ops::close ()
{
  /* Unregister from the event loop.  */
  async (0);
  beneath ()->close ();
}

void
rocm_target_ops::mourn_inferior ()
{
  auto *info = get_rocm_inferior_info ();
  info->has_exited = true;

  if (info->activated)
    amd_dbgapi_deactivated.notify ();
  amd_dbgapi_process_detach (info->process_id);
  info->process_id = AMD_DBGAPI_PROCESS_NONE;

  beneath ()->mourn_inferior ();

  /* FIXME: only unpush on the last activation.  */
  /* Disengage the ROCm target_ops.  */
  unpush_target (&rocm_ops);
}

void
rocm_target_ops::detach (inferior *inf, int from_tty)
{
  auto *info = get_rocm_inferior_info ();

  if (info->activated)
    amd_dbgapi_deactivated.notify ();
  amd_dbgapi_process_detach (info->process_id);
  info->process_id = AMD_DBGAPI_PROCESS_NONE;

  beneath ()->detach (inf, from_tty);
}

void
rocm_target_ops::fetch_registers (struct regcache *regcache, int regno)
{
  struct gdbarch *gdbarch = regcache->arch ();
  struct gdbarch_tdep *tdep = gdbarch_tdep (gdbarch);

  /* delegate to the host routines when not on the device */

  if (!rocm_is_amdgcn_gdbarch (gdbarch))
    {
      beneath ()->fetch_registers (regcache, regno);
      return;
    }

  inferior *inf = find_inferior_ptid (regcache->ptid ());
  amd_dbgapi_process_id_t process_id = get_amd_dbgapi_process_id (inf);
  amd_dbgapi_wave_id_t wave_id = get_amd_dbgapi_wave_id (regcache->ptid ());

  gdb_byte raw[AMDGCN_MAX_REGISTER_SIZE];

  amd_dbgapi_status_t status = amd_dbgapi_read_register (
      process_id, wave_id, tdep->register_ids[regno], 0,
      TYPE_LENGTH (register_type (gdbarch, regno)), raw);

  if (status == AMD_DBGAPI_STATUS_SUCCESS)
    {
      regcache->raw_supply (regno, raw);
    }
  else if (status != AMD_DBGAPI_STATUS_ERROR_INVALID_REGISTER_ID)
    {
      warning (_ ("Couldn't read register %s (#%d)."),
               gdbarch_register_name (gdbarch, regno), regno);
    }
}

void
rocm_target_ops::store_registers (struct regcache *regcache, int regno)
{
  struct gdbarch *gdbarch = regcache->arch ();
  struct gdbarch_tdep *tdep = gdbarch_tdep (gdbarch);
  gdb_byte raw[AMDGCN_MAX_REGISTER_SIZE];

  if (!rocm_is_amdgcn_gdbarch (gdbarch))
    {
      beneath ()->store_registers (regcache, regno);
      return;
    }

  inferior *inf = find_inferior_ptid (regcache->ptid ());
  amd_dbgapi_process_id_t process_id = get_amd_dbgapi_process_id (inf);
  amd_dbgapi_wave_id_t wave_id = get_amd_dbgapi_wave_id (regcache->ptid ());

  regcache->raw_collect (regno, &raw);

  amd_dbgapi_status_t status = amd_dbgapi_write_register (
      process_id, wave_id, tdep->register_ids[regno], 0,
      TYPE_LENGTH (register_type (gdbarch, regno)), raw);

  if (status != AMD_DBGAPI_STATUS_SUCCESS)
    {
      warning (_ ("Couldn't write register %s (#%d)."),
               gdbarch_register_name (gdbarch, regno), regno);
    }
}

/* Fix breakpoints created with an address location while the
   architecture was set to the host (could be fixed in core GDB).  */

static void
rocm_target_breakpoint_fixup (struct breakpoint *b)
{
  if (b->location.get ()
      && event_location_type (b->location.get ()) == ADDRESS_LOCATION
      && gdbarch_bfd_arch_info (b->loc->gdbarch)->arch == bfd_arch_amdgcn
      && gdbarch_bfd_arch_info (b->gdbarch)->arch != bfd_arch_amdgcn)
    {
      b->gdbarch = b->loc->gdbarch;
    }
}

struct gdbarch *
rocm_target_ops::thread_architecture (ptid_t ptid)
{
  static std::result_of<decltype (&ptid_t::tid) (ptid_t)>::type last_tid = 0;
  static struct gdbarch *cached_arch = nullptr;

  if (!ptid_is_gpu (ptid))
    return beneath ()->thread_architecture (ptid);

  /* We can cache the gdbarch for a given wave_id (ptid::tid) because
     wave IDs are unique, and aren't reused.  */
  if (ptid.tid () == last_tid)
    return cached_arch;

  amd_dbgapi_process_id_t process_id = get_amd_dbgapi_process_id ();
  amd_dbgapi_wave_id_t wave_id = get_amd_dbgapi_wave_id (ptid);
  amd_dbgapi_architecture_id_t architecture_id;

  if (amd_dbgapi_wave_get_info (process_id, wave_id,
                                AMD_DBGAPI_WAVE_INFO_ARCHITECTURE,
                                sizeof (architecture_id), &architecture_id)
      != AMD_DBGAPI_STATUS_SUCCESS)
    error (_ ("Couldn't get architecture for wave_%ld"), ptid.tid ());

  uint32_t elf_amdgpu_machine;
  if (amd_dbgapi_architecture_get_info (
          architecture_id, AMD_DBGAPI_ARCHITECTURE_INFO_ELF_AMDGPU_MACHINE,
          sizeof (elf_amdgpu_machine), &elf_amdgpu_machine)
      != AMD_DBGAPI_STATUS_SUCCESS)
    error (_ ("Couldn't get elf_amdgpu_machine for architecture_%ld"),
           architecture_id.handle);

  struct gdbarch_info info;
  gdbarch_info_init (&info);

  info.bfd_arch_info = bfd_lookup_arch (bfd_arch_amdgcn, elf_amdgpu_machine);
  info.byte_order = BFD_ENDIAN_LITTLE;
  info.osabi = GDB_OSABI_AMDGPU_HSA;

  last_tid = ptid.tid ();
  if (!(cached_arch = gdbarch_find_by_info (info)))
    error (_ ("Couldn't get elf_amdgpu_machine (%#x)"), elf_amdgpu_machine);

  return cached_arch;
}

void
rocm_target_ops::update_thread_list ()
{
  for (inferior *inf : all_inferiors ())
    {
      amd_dbgapi_process_id_t process_id;
      amd_dbgapi_wave_id_t *wave_list;
      size_t count;

      process_id = get_amd_dbgapi_process_id (inf);
      if (process_id.handle == AMD_DBGAPI_PROCESS_NONE.handle)
        {
          /* The inferior may not be attached yet.  */
          continue;
        }

      amd_dbgapi_changed_t changed;
      amd_dbgapi_status_t status;
      if ((status
           = amd_dbgapi_wave_list (process_id, &count, &wave_list, &changed))
          != AMD_DBGAPI_STATUS_SUCCESS)
        error (_ ("amd_dbgapi_wave_list failed (rc=%d)"), status);

      if (changed == AMD_DBGAPI_CHANGED_NO)
        continue;

      /* Create a set and free the wave list.  */
      std::set<std::result_of<decltype (&ptid_t::tid) (ptid_t)>::type> threads;
      for (size_t i = 0; i < count; ++i)
        threads.emplace (wave_list[i].handle);
      xfree (wave_list);

      /* Then prune the wave_ids that already have a thread_info.  */
      for (thread_info *tp : inf->non_exited_threads ())
        if (ptid_is_gpu (tp->ptid))
          threads.erase (tp->ptid.tid ());

      /* The wave_ids that are left require a new thread_info.  */
      for (auto &&tid : threads)
        {
          ptid_t wave_ptid (inf->pid, 1, tid);
          /* FIXME: is this really needed?
          amd_dbgapi_wave_state_t state;

          if (amd_dbgapi_wave_get_info (
                  process_id, get_amd_dbgapi_wave_id (wave_ptid),
                  AMD_DBGAPI_WAVE_INFO_STATE, sizeof (state), &state)
              != AMD_DBGAPI_STATUS_SUCCESS)
            continue;*/

          add_thread_silent (wave_ptid);
          set_running (wave_ptid, 1);
          set_executing (wave_ptid, 1);
        }
    }

  /* Give the beneath target a chance to do extra processing.  */
  this->beneath ()->update_thread_list ();
}

static void
rocm_target_solib_loaded (struct so_list *solib)
{
  std::string so_name (solib->so_name);

  auto pos = so_name.find_last_of ('/');
  std::string library_name
      = so_name.substr (pos == std::string::npos ? 0 : (pos + 1));

  /* Notify the amd_dbgapi that a shared library has been loaded.  */
  for (auto &&value : get_rocm_inferior_info ()->notify_solib_map)
    {
      if (!value.second.solib && library_name == value.second.compare)
        {
          value.second.solib = solib;

          amd_dbgapi_report_shared_library (
              get_amd_dbgapi_process_id (),
              amd_dbgapi_shared_library_id_t{ value.first },
              AMD_DBGAPI_SHARED_LIBRARY_STATE_LOADED);

          /* The rocm target may not be engaged yet, we need to process the
            events now in case there is a runtime event pending.  */
          rocm_process_event_queue ();
        }
    }
}

static void
rocm_target_solib_unloaded (struct so_list *solib)
{
  /* Notify the amd_dbgapi that a shared library will unload.  */
  for (auto &&value : get_rocm_inferior_info ()->notify_solib_map)
    /* TODO: If we want to support file name wildcards, change this code.  */
    if (solib == value.second.solib)
      {
        struct rocm_inferior_info *info = get_rocm_inferior_info ();

        amd_dbgapi_report_shared_library (
            info->process_id, amd_dbgapi_shared_library_id_t{ value.first },
            AMD_DBGAPI_SHARED_LIBRARY_STATE_UNLOADED);

        /* Delete breakpoints that were left inserted in this shared library.
         */
        for (auto it = info->breakpoint_map.begin ();
             it != info->breakpoint_map.end ();)
          if (solib_contains_address_p (solib, it->second->loc->address))
            {
              warning (_ ("breakpoint_%ld is still inserted after "
                          "shared_library_%ld was unloaded"),
                       it->first, value.first);
              delete_breakpoint (it->second);
              it = info->breakpoint_map.erase (it);
            }
          else
            ++it;

        value.second.solib = nullptr;
      }
}

static void
rocm_target_inferior_created (struct target_ops *target, int from_tty)
{
  struct inferior *inf = current_inferior ();
  auto *info = get_rocm_inferior_info (inf);
  amd_dbgapi_status_t status;

  if (!target_can_async_p ())
    {
      warning (_ ("ROCgdb requires target-async, GPU debugging is disabled"));
      return;
    }

  gdb_assert (info->wave_stop_events.empty ());

  status = amd_dbgapi_process_attach (
      reinterpret_cast<amd_dbgapi_client_process_id_t> (inf),
      &info->process_id);

  if (status != AMD_DBGAPI_STATUS_SUCCESS)
    error (_ ("Could not attach to process %d"), inf->pid);

  if (amd_dbgapi_process_get_info (info->process_id,
                                   AMD_DBGAPI_PROCESS_INFO_NOTIFIER,
                                   sizeof (info->notifier), &info->notifier)
      != AMD_DBGAPI_STATUS_SUCCESS)
    {
      warning (_ ("Could not retrieve process %d's notifier"), inf->pid);
      amd_dbgapi_process_detach (info->process_id);
      return;
    }

  /* We add a file handler for events returned by the debugger api. We'll use
     this handler to signal our async handler that events are available.  */
  add_file_handler (
      info->notifier,
      [] (int error, gdb_client_data client_data) {
        auto info_ = static_cast<struct rocm_inferior_info *> (client_data);
        int ret;

        /* Drain the notifier pipe.  */
        do
          {
            char buf;
            ret = read (info_->notifier, &buf, 1);
          }
        while (ret >= 0 || (ret == -1 && errno == EINTR));

        /* Signal our async handler.  */
        async_file_mark ();
      },
      info);

  /* Attaching to the inferior may have generated runtime events, process
     them now.  */
  rocm_process_event_queue ();
}

static void
rocm_target_inferior_exit (struct inferior *inf)
{
  auto *info = get_rocm_inferior_info (inf);

  if (info->notifier != -1)
    delete_file_handler (info->notifier);

  if (info->process_id.handle != AMD_DBGAPI_PROCESS_NONE.handle)
    amd_dbgapi_process_detach (info->process_id);

  /* Delete the breakpoints that are still active.  */
  for (auto &&value : info->breakpoint_map)
    delete_breakpoint (value.second);

  rocm_inferior_data.clear (inf);
}

static cli_style_option warning_style ("rocm_warning", ui_file_style::RED);
static cli_style_option info_style ("rocm_info", ui_file_style::GREEN);
static cli_style_option verbose_style ("rocm_verbose", ui_file_style::BLUE);

static amd_dbgapi_callbacks_t dbgapi_callbacks = {
  /* allocate_memory.  */
  .allocate_memory = malloc,

  /* deallocate_memory.  */
  .deallocate_memory = free,

  /* get_os_pid.  */
  .get_os_pid = [] (amd_dbgapi_client_process_id_t client_process_id,
                    pid_t *pid) -> amd_dbgapi_status_t {
    inferior *inf = reinterpret_cast<inferior *> (client_process_id);
    struct rocm_inferior_info *info = get_rocm_inferior_info (inf);

    if (info->has_exited)
      return AMD_DBGAPI_STATUS_ERROR_PROCESS_EXITED;

    *pid = inf->pid;
    return AMD_DBGAPI_STATUS_SUCCESS;
  },

  /* enable_notify_shared_library callback.  */
  .enable_notify_shared_library
  = [] (amd_dbgapi_client_process_id_t client_process_id,
        const char *library_name, amd_dbgapi_shared_library_id_t library_id,
        amd_dbgapi_shared_library_state_t *library_state)
      -> amd_dbgapi_status_t {
    inferior *inf = reinterpret_cast<inferior *> (client_process_id);
    struct rocm_inferior_info *info = get_rocm_inferior_info (inf);

    if (!library_name || !library_state)
      return AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT;

    /* Check that the library_name is valid.  If must not be empty, and
       should not have wildcard characters.  */
    if (*library_name == '\0'
        || std::string (library_name).find_first_of ("*?[]")
               != std::string::npos)
      return AMD_DBGAPI_STATUS_ERROR_INVALID_ARGUMENT;

    /* Check whether the library is already loaded.  */
    struct so_list *solib;
    for (solib = inf->pspace->so_list; solib; solib = solib->next)
      {
        std::string so_name (solib->so_name);

        auto pos = so_name.find_last_of ('/');
        if (so_name.substr (pos == std::string::npos ? 0 : (pos + 1))
            == library_name)
          break;
      }

    /* Add a new entry in the notify_solib_map.  */
    if (!info->notify_solib_map
             .emplace (std::piecewise_construct,
                       std::forward_as_tuple (library_id.handle),
                       std::forward_as_tuple (rocm_notify_shared_library_info{
                           library_name, solib }))
             .second)
      return AMD_DBGAPI_STATUS_ERROR;

    *library_state = solib != nullptr
                         ? AMD_DBGAPI_SHARED_LIBRARY_STATE_LOADED
                         : AMD_DBGAPI_SHARED_LIBRARY_STATE_UNLOADED;

    return AMD_DBGAPI_STATUS_SUCCESS;
  },

  /* disable_notify_shared_library callback.  */
  .disable_notify_shared_library
  = [] (amd_dbgapi_client_process_id_t client_process_id,
        amd_dbgapi_shared_library_id_t library_id) -> amd_dbgapi_status_t {
    inferior *inf = reinterpret_cast<inferior *> (client_process_id);
    struct rocm_inferior_info *info = get_rocm_inferior_info (inf);

    auto it = info->notify_solib_map.find (library_id.handle);
    if (it == info->notify_solib_map.end ())
      return AMD_DBGAPI_STATUS_ERROR_INVALID_SHARED_LIBRARY_ID;

    info->notify_solib_map.erase (it);
    return AMD_DBGAPI_STATUS_SUCCESS;
  },

  /* get_symbol_address callback.  */
  .get_symbol_address =
      [] (amd_dbgapi_client_process_id_t client_process_id,
          amd_dbgapi_shared_library_id_t library_id, const char *symbol_name,
          amd_dbgapi_global_address_t *address) {
        inferior *inf = reinterpret_cast<inferior *> (client_process_id);
        struct rocm_inferior_info *info = get_rocm_inferior_info (inf);

        auto it = info->notify_solib_map.find (library_id.handle);
        if (it == info->notify_solib_map.end ())
          return AMD_DBGAPI_STATUS_ERROR_INVALID_SHARED_LIBRARY_ID;

        struct so_list *solib = it->second.solib;
        if (!solib)
          return AMD_DBGAPI_STATUS_ERROR_LIBRARY_NOT_LOADED;

        solib_read_symbols (solib, 0);
        gdb_assert (solib->objfile);

        struct bound_minimal_symbol msymbol
            = lookup_minimal_symbol (symbol_name, NULL, solib->objfile);

        if (!msymbol.minsym || BMSYMBOL_VALUE_ADDRESS (msymbol) == 0)
          return AMD_DBGAPI_STATUS_ERROR_SYMBOL_NOT_FOUND;

        *address = BMSYMBOL_VALUE_ADDRESS (msymbol);
        return AMD_DBGAPI_STATUS_SUCCESS;
      },

  /* set_breakpoint callback.  */
  .insert_breakpoint =
      [] (amd_dbgapi_client_process_id_t client_process_id,
          amd_dbgapi_shared_library_id_t shared_library_id,
          amd_dbgapi_global_address_t address,
          amd_dbgapi_breakpoint_id_t breakpoint_id) {
        inferior *inf = reinterpret_cast<inferior *> (client_process_id);
        struct rocm_inferior_info *info = get_rocm_inferior_info (inf);

        /* Initialize the breakpoint ops lazily since we depend on
           bkpt_breakpoint_ops and we can't control the order in which
           initializers are called.  */
        if (rocm_breakpoint_ops.check_status == NULL)
          {
            rocm_breakpoint_ops = bkpt_breakpoint_ops;
            rocm_breakpoint_ops.check_status = rocm_breakpoint_check_status;
            rocm_breakpoint_ops.re_set = rocm_breakpoint_re_set;
          }

        auto it = info->breakpoint_map.find (breakpoint_id.handle);
        if (it != info->breakpoint_map.end ())
          return AMD_DBGAPI_STATUS_ERROR_INVALID_BREAKPOINT_ID;

        /* Create a new breakpoint.  */
        struct obj_section *section = find_pc_section (address);
        if (!section || !section->objfile)
          return AMD_DBGAPI_STATUS_ERROR;

        event_location_up location
            = new_address_location (address, nullptr, 0);
        if (!create_breakpoint (
                get_objfile_arch (section->objfile), location.get (),
                /*cond_string*/ NULL, /*thread*/ -1, /*extra_sring*/ NULL,
                /*parse_extra*/ 0, /*tempflag*/ 0, /*bptype*/ bp_breakpoint,
                /*ignore_count*/ 0, /*pending_break*/ AUTO_BOOLEAN_FALSE,
                /*ops*/ &rocm_breakpoint_ops, /*from_tty*/ 0,
                /*enabled*/ 1, /*internal*/ 1, /*flags*/ 0))
          return AMD_DBGAPI_STATUS_ERROR;

        /* Find our breakpoint in the breakpoint list.  */
        auto bp_loc = std::make_pair (inf->aspace, address);
        auto bp = breakpoint_find_if (
            [] (struct breakpoint *b, void *data) {
              auto *arg = static_cast<decltype (&bp_loc)> (data);
              if (b->ops == &rocm_breakpoint_ops && b->loc
                  && b->loc->pspace->aspace == arg->first
                  && b->loc->address == arg->second)
                return 1;
              return 0;
            },
            reinterpret_cast<void *> (&bp_loc));

        if (!bp)
          error (_ ("Could not find breakpoint"));

        info->breakpoint_map.emplace (breakpoint_id.handle, bp);
        return AMD_DBGAPI_STATUS_SUCCESS;
      },

  /* remove_breakpoint callback.  */
  .remove_breakpoint =
      [] (amd_dbgapi_client_process_id_t client_process_id,
          amd_dbgapi_breakpoint_id_t breakpoint_id) {
        inferior *inf = reinterpret_cast<inferior *> (client_process_id);
        struct rocm_inferior_info *info = get_rocm_inferior_info (inf);

        auto it = info->breakpoint_map.find (breakpoint_id.handle);
        if (it == info->breakpoint_map.end ())
          return AMD_DBGAPI_STATUS_ERROR_INVALID_BREAKPOINT_ID;

        delete_breakpoint (it->second);
        info->breakpoint_map.erase (it);

        return AMD_DBGAPI_STATUS_SUCCESS;
      },

  .log_message
  = [] (amd_dbgapi_log_level_t level, const char *message) -> void {
    gdb::optional<target_terminal::scoped_restore_terminal_state> tstate;

    if (level > get_debug_amdgpu_log_level ())
      return;

    if (target_supports_terminal_ours ())
      {
        tstate.emplace ();
        target_terminal::ours_for_output ();
      }

    if (filtered_printing_initialized ())
      wrap_here ("");

    struct ui_file *out_file
        = (level >= AMD_DBGAPI_LOG_LEVEL_INFO) ? gdb_stdlog : gdb_stderr;

    switch (level)
      {
      case AMD_DBGAPI_LOG_LEVEL_FATAL_ERROR:
        fputs_unfiltered ("amd-dbgapi: ", out_file);
        break;
      case AMD_DBGAPI_LOG_LEVEL_WARNING:
        fputs_styled ("amd-dbgapi: ", warning_style.style (), out_file);
        break;
      case AMD_DBGAPI_LOG_LEVEL_INFO:
        fputs_styled ("amd-dbgapi: ", info_style.style (), out_file);
        break;
      case AMD_DBGAPI_LOG_LEVEL_VERBOSE:
        fputs_styled ("amd-dbgapi: ", verbose_style.style (), out_file);
        break;
      }

    fputs_unfiltered (message, out_file);
    fputs_unfiltered ("\n", out_file);
  }
};

/* Implementation of `_wave_id' variable.  */

static struct value *
rocm_wave_id_make_value (struct gdbarch *gdbarch, struct internalvar *var,
                         void *ignore)
{
  if (ptid_is_gpu (inferior_ptid))
    {
      amd_dbgapi_process_id_t process_id = get_amd_dbgapi_process_id ();
      amd_dbgapi_wave_id_t wave_id = get_amd_dbgapi_wave_id (inferior_ptid);
      uint32_t group_ids[3], wave_in_group;

      if (amd_dbgapi_wave_get_info (process_id, wave_id,
                                    AMD_DBGAPI_WAVE_INFO_WORK_GROUP_COORD,
                                    sizeof (group_ids), &group_ids)
              == AMD_DBGAPI_STATUS_SUCCESS
          && amd_dbgapi_wave_get_info (
                 process_id, wave_id,
                 AMD_DBGAPI_WAVE_INFO_WAVE_NUMBER_IN_WORK_GROUP,
                 sizeof (wave_in_group), &wave_in_group)
                 == AMD_DBGAPI_STATUS_SUCCESS)
        {
          std::string wave_id_str
              = string_printf ("(%d,%d,%d)/%d", group_ids[2], group_ids[1],
                               group_ids[0], wave_in_group);

          return value_cstring (wave_id_str.data (), wave_id_str.length () + 1,
                                builtin_type (gdbarch)->builtin_char);
        }
    }

  return allocate_value (builtin_type (gdbarch)->builtin_void);
}

static const struct internalvar_funcs rocm_wave_id_funcs
    = { rocm_wave_id_make_value, NULL, NULL };

/* List of set/show debug amd_dbgapi commands.  */
struct cmd_list_element *set_debug_amdgpu_list;
struct cmd_list_element *show_debug_amdgpu_list;

static void
set_debug_amdgpu (const char *arg, int from_tty)
{
  help_list (set_debug_amdgpu_list, "set debug amd-dbgapi ",
             (enum command_class) - 1, gdb_stdout);
}

static void
show_debug_amdgpu (const char *args, int from_tty)
{
  cmd_show_list (show_debug_amdgpu_list, from_tty, "");
}

constexpr char amd_dbgapi_log_level_off[] = "off";
constexpr char amd_dbgapi_log_level_error[] = "error";
constexpr char amd_dbgapi_log_level_warning[] = "warning";
constexpr char amd_dbgapi_log_level_info[] = "info";
constexpr char amd_dbgapi_log_level_verbose[] = "verbose";

constexpr const char *debug_amdgpu_log_level_enums[]
    = { [AMD_DBGAPI_LOG_LEVEL_NONE] = amd_dbgapi_log_level_off,
        [AMD_DBGAPI_LOG_LEVEL_FATAL_ERROR] = amd_dbgapi_log_level_error,
        [AMD_DBGAPI_LOG_LEVEL_WARNING] = amd_dbgapi_log_level_warning,
        [AMD_DBGAPI_LOG_LEVEL_INFO] = amd_dbgapi_log_level_info,
        [AMD_DBGAPI_LOG_LEVEL_VERBOSE] = amd_dbgapi_log_level_verbose,
        nullptr };

static const char *debug_amdgpu_log_level = amd_dbgapi_log_level_warning;

static amd_dbgapi_log_level_t
get_debug_amdgpu_log_level ()
{
  size_t pos;
  for (pos = 0; debug_amdgpu_log_level_enums[pos]; ++pos)
    if (debug_amdgpu_log_level == debug_amdgpu_log_level_enums[pos])
      break;

  gdb_assert (debug_amdgpu_log_level_enums[pos]);
  return static_cast<amd_dbgapi_log_level_t> (pos);
}

static void
set_debug_amdgpu_log_level (const char *args, int from_tty,
                                struct cmd_list_element *c)
{
  amd_dbgapi_set_log_level (get_debug_amdgpu_log_level ());
}

static void
show_debug_amdgpu_log_level (struct ui_file *file, int from_tty,
                                 struct cmd_list_element *c, const char *value)
{
  fprintf_filtered (file, _ ("The amdgpu log level is %s.\n"), value);
}

static void
info_agents_command (const char *args, int from_tty)
{
  amd_dbgapi_process_id_t process_id = get_amd_dbgapi_process_id ();
  struct ui_out *uiout = current_uiout;
  amd_dbgapi_status_t status;

  amd_dbgapi_agent_id_t *agent_list;
  size_t count = 0;

  if (process_id.handle != AMD_DBGAPI_PROCESS_NONE.handle
      && (status
          = amd_dbgapi_agent_list (process_id, &count, &agent_list, nullptr))
             != AMD_DBGAPI_STATUS_SUCCESS)
    error (_ ("amd_dbgapi_agent_list failed (rc=%d)"), status);

  if (!count && !uiout->is_mi_like_p ())
    {
      uiout->field_string (NULL,
                           _ ("No agents are currently active.\n"));
      return;
    }

  std::sort (&agent_list[0], &agent_list[count],
             [] (amd_dbgapi_agent_id_t lhs, amd_dbgapi_agent_id_t rhs) {
               return lhs.handle < rhs.handle;
             });

  amd_dbgapi_wave_id_t wave_id = get_amd_dbgapi_wave_id (inferior_ptid);
  amd_dbgapi_agent_id_t current_agent_id;

  if (amd_dbgapi_wave_get_info (process_id, wave_id,
                                AMD_DBGAPI_WAVE_INFO_AGENT,
                                sizeof (current_agent_id), &current_agent_id)
      != AMD_DBGAPI_STATUS_SUCCESS)
    current_agent_id = AMD_DBGAPI_AGENT_NONE;

  /* Calculate the maximum size needed to print the agents names.  */
  std::vector<std::string> agent_names (count);

  size_t max_name_len = 0;
  for (size_t i = 0; i < count; ++i)
    {
      char *agent_name;

      if ((status = amd_dbgapi_agent_get_info (
               process_id, agent_list[i], AMD_DBGAPI_AGENT_INFO_NAME,
               sizeof (agent_name), &agent_name))
          != AMD_DBGAPI_STATUS_SUCCESS)
        {
          if (status == AMD_DBGAPI_STATUS_ERROR_INVALID_AGENT_ID)
            agent_names[i] = "N/A";
          else
            error (_ ("amd_dbgapi_agent_get_info failed (rc=%d)"), status);
        }
      else
        {
          agent_names[i] = agent_name;
          xfree (agent_name);
        }

      max_name_len = std::max (max_name_len, agent_names[i].size ());
    }

  /* Header:  */
  ui_out_emit_table table_emmitter (uiout, 7, count, "InfoRocmDevicesTable");

  uiout->table_header (1, ui_left, "current", "");
  uiout->table_header (2, ui_left, "agent_id", "Id");
  uiout->table_header (9, ui_left, "target_id", "Target Id");
  uiout->table_header (std::max (11ul, max_name_len), ui_left, "name",
                       "Device Name");
  uiout->table_header (5, ui_left, "cores", "Cores");
  uiout->table_header (7, ui_left, "threads", "Threads");
  uiout->table_header (8, ui_left, "location_id", "PCI Slot");
  uiout->table_body ();

  /* Rows:  */
  for (size_t i = 0; i < count; ++i)
    {
      ui_out_emit_tuple tuple_emitter (uiout, "InfoRocmDevicesRow");

      /* current  */
      if (agent_list[i].handle == current_agent_id.handle)
        uiout->field_string ("current", "*");
      else
        uiout->field_skip ("current");

      /* agent_id  */
      uiout->field_signed ("agent_id", agent_list[i].handle);

      /* target_id  */
      amd_dbgapi_os_agent_id_t target_id;
      if ((status = amd_dbgapi_agent_get_info (process_id, agent_list[i],
                                               AMD_DBGAPI_AGENT_INFO_OS_ID,
                                               sizeof (target_id), &target_id))
          != AMD_DBGAPI_STATUS_SUCCESS)
        error (_ ("amd_dbgapi_agent_get_info failed (rc=%d)"), status);
      else
        uiout->field_signed ("target_id", target_id);

      /* name  */
      uiout->field_string ("name", agent_names[i]);

      /* cores  */
      size_t cores;
      if ((status = amd_dbgapi_agent_get_info (
               process_id, agent_list[i],
               AMD_DBGAPI_AGENT_INFO_EXECUTION_UNIT_COUNT, sizeof (cores),
               &cores))
          != AMD_DBGAPI_STATUS_SUCCESS)
        error (_ ("amd_dbgapi_agent_get_info failed (rc=%d)"), status);
      else
        uiout->field_signed ("cores", cores);

      /* threads  */
      size_t threads;
      if ((status = amd_dbgapi_agent_get_info (
               process_id, agent_list[i],
               AMD_DBGAPI_AGENT_INFO_MAX_WAVES_PER_EXECUTION_UNIT,
               sizeof (threads), &threads))
          != AMD_DBGAPI_STATUS_SUCCESS)
        error (_ ("amd_dbgapi_agent_get_info failed (rc=%d)"), status);
      else
        uiout->field_signed ("threads", cores * threads);

      /* location  */
      uint16_t location_id;
      if ((status = amd_dbgapi_agent_get_info (
               process_id, agent_list[i], AMD_DBGAPI_AGENT_INFO_PCI_SLOT,
               sizeof (location_id), &location_id))
          != AMD_DBGAPI_STATUS_SUCCESS)
        error (_ ("amd_dbgapi_agent_get_info failed (rc=%d)"), status);
      else
        uiout->field_string (
            "location_id",
            string_printf ("%02x:%02x.%d", (location_id >> 8) & 0xFF,
                           (location_id >> 3) & 0x1F, location_id & 0x7));

      uiout->text ("\n");
    }

  xfree (agent_list);
  gdb_flush (gdb_stdout);
}

/* -Wmissing-prototypes */
extern initialize_file_ftype _initialize_rocm_tdep;

void
_initialize_rocm_tdep (void)
{
  /* Make sure the loaded debugger library version is greater than or equal to
     the one used to build ROCgdb.  */
  uint32_t major, minor, patch;
  amd_dbgapi_get_version (&major, &minor, &patch);
  if (major != AMD_DBGAPI_VERSION_MAJOR || minor < AMD_DBGAPI_VERSION_MINOR)
    error (
        _ ("amd-dbgapi library version mismatch, got %d.%d.%d, need %d.%d+"),
        major, minor, patch, AMD_DBGAPI_VERSION_MAJOR,
        AMD_DBGAPI_VERSION_MINOR);

  /* Initialize the ROCm Debug API.  */
  amd_dbgapi_status_t status = amd_dbgapi_initialize (&dbgapi_callbacks);
  if (status != AMD_DBGAPI_STATUS_SUCCESS)
    error (_ ("amd-dbgapi failed to initialize (rc=%d)"), status);

  /* Set the initial log level.  */
  amd_dbgapi_set_log_level (get_debug_amdgpu_log_level ());

  /* Install observers.  */
  gdb::observers::breakpoint_created.attach (rocm_target_breakpoint_fixup);
  gdb::observers::solib_loaded.attach (rocm_target_solib_loaded);
  gdb::observers::solib_unloaded.attach (rocm_target_solib_unloaded);
  gdb::observers::inferior_created.attach (rocm_target_inferior_created);
  gdb::observers::inferior_exit.attach (rocm_target_inferior_exit);

  amd_dbgapi_activated.attach (rocm_target_dbgapi_activated);
  amd_dbgapi_deactivated.attach (rocm_target_dbgapi_deactivated);

  create_internalvar_type_lazy ("_wave_id", &rocm_wave_id_funcs, NULL);

  add_prefix_cmd (
      "amdgpu", no_class, set_debug_amdgpu,
      _ ("Generic command for setting amdgpu debugging flags"),
      &set_debug_amdgpu_list, "set debug amdgpu ", 0, &setdebuglist);

  add_prefix_cmd (
      "amdgpu", no_class, show_debug_amdgpu,
      _ ("Generic command for showing amdgpu debugging flags"),
      &show_debug_amdgpu_list, "show debug amdgpu ", 0,
      &showdebuglist);

  add_setshow_enum_cmd (
      "log-level", class_maintenance, debug_amdgpu_log_level_enums,
      &debug_amdgpu_log_level, _ ("Set the amdgpu log level."),
      _ ("Show the amdgpu log level."),
      _ ("off     == no logging is enabled\n"
         "error   == fatal errors are reported\n"
         "warning == fatal errors and warnings are reported\n"
         "info    == fatal errors, warnings, and info messages are reported\n"
         "verbose == all messages are reported"),
      set_debug_amdgpu_log_level, show_debug_amdgpu_log_level,
      &set_debug_amdgpu_list, &show_debug_amdgpu_list);

  add_cmd ("agents", class_info, info_agents_command,
           _ ("Info about currently active agents."), &infolist);
}
