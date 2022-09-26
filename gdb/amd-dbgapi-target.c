/* Target used to communicate with the AMD Debugger API.

   Copyright (C) 2019-2022 Free Software Foundation, Inc.
   Copyright (C) 2019-2022 Advanced Micro Devices, Inc. All rights reserved.

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

#include "amdgpu-tdep.h"
#include "arch-utils.h"
#include "async-event.h"
#include "cli/cli-style.h"
#include "demangle.h"
#include "displaced-stepping.h"
#include "environ.h"
#include "filenames.h"
#include "gdb-demangle.h"
#include "gdbcmd.h"
#include "gdbcore.h"
#include "gdbsupport/event-loop.h"
#include "gdbsupport/filestuff.h"
#include "gdbsupport/gdb_unique_ptr.h"
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
#include "amd-dbgapi-target.h"
#include "solib.h"
#include "solist.h"
#include "symfile.h"
#include "tid-parse.h"
#include "cli/cli-decode.h"

#include <dlfcn.h>
#include <list>
#include <map>
#include <set>
#include <sstream>
#include <unordered_map>

#include <signal.h>
#include <stdarg.h>

#include <amd-dbgapi/amd-dbgapi.h>

/* Big enough to hold the size of the largest register in bytes.  */
#define AMDGPU_MAX_REGISTER_SIZE 256

/* amd-dbgapi-specific inferior data.  */

struct amd_dbgapi_inferior_info
{
  explicit amd_dbgapi_inferior_info (inferior *inf) : inf (inf) {}

  /* Backlink to inferior.  */
  inferior *inf;

  /* The amd_dbgapi_process_id for this inferior.  */
  amd_dbgapi_process_id_t process_id{ AMD_DBGAPI_PROCESS_NONE };

  /* The amd_dbgapi_notifier_t for this inferior.  */
  amd_dbgapi_notifier_t notifier{ -1 };

  /* The status of the inferior's runtime support.  */
  amd_dbgapi_runtime_state_t runtime_state{
    AMD_DBGAPI_RUNTIME_STATE_UNLOADED
  };

  /* This value mirrors the current forward progress needed value for this
     process in dbgapi.  It is used to avoid unnecessary calls to
     amd_dbgapi_process_set_progress, to reduce the noise in the logs dbgapi
     logs

     Initialized to true, since that's the default in dbgapi too.  */
  bool forward_progress_required = true;

  struct
  {
    /* Whether precise memory reporting is requested.  */
    bool requested{ false };
    /* Precise memory was requested and successfully enabled by the dbgapi.  */
    bool enabled{ false };
  } precise_memory;

  std::unordered_map<decltype (amd_dbgapi_breakpoint_id_t::handle),
		     struct breakpoint *>
    breakpoint_map;

  std::map<CORE_ADDR, std::pair<CORE_ADDR, amd_dbgapi_watchpoint_id_t>>
    watchpoint_map;

  /* List of pending events the amd-dbgapi target retrieved from the dbgapi.  */
  std::list<std::pair<ptid_t, target_waitstatus>> wave_events;

  std::unordered_map<thread_info *,
		     decltype (amd_dbgapi_displaced_stepping_id_t::handle)>
    stepping_id_map;
};

static amd_dbgapi_event_id_t process_event_queue (
  amd_dbgapi_process_id_t process_id = AMD_DBGAPI_PROCESS_NONE,
  amd_dbgapi_event_kind_t until_event_kind = AMD_DBGAPI_EVENT_KIND_NONE);

/* Return the inferior's amd_dbgapi_inferior_info struct.  */
static struct amd_dbgapi_inferior_info *
get_amd_dbgapi_inferior_info (struct inferior *inferior = nullptr);

static const target_info amd_dbgapi_target_info = {
  "amd-dbgapi",
  N_("AMD Debugger API"),
  N_("GPU debugging using the AMD Debugger API")
};

static amd_dbgapi_log_level_t get_debug_amdgpu_log_level ();

struct amd_dbgapi_target final : public target_ops
{
  bool report_thread_events = false;

  const target_info &
  info () const override
  {
    return amd_dbgapi_target_info;
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

  bool has_pending_events () override;
  ptid_t wait (ptid_t, struct target_waitstatus *, target_wait_flags) override;
  void resume (ptid_t, int, enum gdb_signal) override;
  void commit_resumed () override;
  void stop (ptid_t ptid) override;

  void fetch_registers (struct regcache *, int) override;
  void store_registers (struct regcache *, int) override;

  void update_thread_list () override;

  struct gdbarch *thread_architecture (ptid_t) override;

  void
  thread_events (int enable) override
  {
    report_thread_events = enable;
    beneath ()->thread_events (enable);
  }

  std::string pid_to_str (ptid_t ptid) override;

  std::string lane_to_str (thread_info *thr, int lane) override;

  std::string dispatch_pos_str (thread_info *thr) override;
  std::string thread_workgroup_pos_str (thread_info *thr) override;
  std::string lane_workgroup_pos_str (thread_info *thr, int lane) override;

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

  bool
  supports_displaced_step (thread_info *thread) override
  {
    if (!ptid_is_gpu (thread->ptid))
      return beneath ()->supports_displaced_step (thread);
    return true;
  }

  displaced_step_prepare_status
  displaced_step_prepare (thread_info *thread,
			  CORE_ADDR &displaced_pc) override;

  displaced_step_finish_status displaced_step_finish (thread_info *thread,
						      gdb_signal sig) override;

  void follow_exec (inferior *follow_inf, ptid_t ptid,
		    const char *execd_pathname) override;

  void follow_fork (inferior *child_inf, ptid_t child_ptid,
		    target_waitkind fork_kind, bool follow_child,
		    bool detach_fork) override;
  void prevent_new_threads (bool prevent, inferior *inf) override;
};

static struct amd_dbgapi_target the_amd_dbgapi_target;

/* amd-dbgapi breakpoint ops.  */
static struct breakpoint_ops amd_dbgapi_target_breakpoint_ops;

/* Per-inferior data key.  */
static const struct inferior_key<amd_dbgapi_inferior_info>
  amd_dbgapi_inferior_data;

/* The async event handler registered with the event loop, indicating that we
   might have events to report to the core and that we'd like our wait method
   to be called.

   This is nullptr when async is disabled and non-nullptr when async is
   enabled.  */
static async_event_handler *amd_dbgapi_async_event_handler = nullptr;

/* Return the target id string for a given wave.  */
static std::string
wave_target_id_string (amd_dbgapi_wave_id_t wave_id)
{
  amd_dbgapi_dispatch_id_t dispatch_id;
  amd_dbgapi_queue_id_t queue_id;
  amd_dbgapi_agent_id_t agent_id;
  uint32_t group_ids[3], wave_in_group;

  std::string str = "AMDGPU Wave";

  str += (amd_dbgapi_wave_get_info (wave_id, AMD_DBGAPI_WAVE_INFO_AGENT,
				    sizeof (agent_id), &agent_id)
	  == AMD_DBGAPI_STATUS_SUCCESS)
	   ? string_printf (" %ld", agent_id.handle)
	   : " ?";

  str += (amd_dbgapi_wave_get_info (wave_id, AMD_DBGAPI_WAVE_INFO_QUEUE,
				    sizeof (queue_id), &queue_id)
	  == AMD_DBGAPI_STATUS_SUCCESS)
	   ? string_printf (":%ld", queue_id.handle)
	   : ":?";

  str += (amd_dbgapi_wave_get_info (wave_id, AMD_DBGAPI_WAVE_INFO_DISPATCH,
				    sizeof (dispatch_id), &dispatch_id)
	  == AMD_DBGAPI_STATUS_SUCCESS)
	   ? string_printf (":%ld", dispatch_id.handle)
	   : ":?";

  str += string_printf (":%ld", wave_id.handle);

  str += amd_dbgapi_wave_get_info (wave_id,
				   AMD_DBGAPI_WAVE_INFO_WORKGROUP_COORD,
				   sizeof (group_ids), &group_ids)
	     == AMD_DBGAPI_STATUS_SUCCESS
	   ? string_printf (" (%d,%d,%d)", group_ids[0], group_ids[1],
			    group_ids[2])
	   : " (?,?,?)";

  str += amd_dbgapi_wave_get_info (
	   wave_id, AMD_DBGAPI_WAVE_INFO_WAVE_NUMBER_IN_WORKGROUP,
	   sizeof (wave_in_group), &wave_in_group)
	     == AMD_DBGAPI_STATUS_SUCCESS
	   ? string_printf ("/%d", wave_in_group)
	   : "/?";

  return str;
}

/* Return the target id string for a given dispatch.  */
static std::string
dispatch_target_id_string (amd_dbgapi_dispatch_id_t dispatch_id)
{
  amd_dbgapi_queue_id_t queue_id;
  amd_dbgapi_status_t status
    = amd_dbgapi_dispatch_get_info (dispatch_id,
				    AMD_DBGAPI_DISPATCH_INFO_QUEUE,
				    sizeof (queue_id), &queue_id);
  if (status != AMD_DBGAPI_STATUS_SUCCESS)
    error (_("amd_dbgapi_dispatch_get_info failed to get queue id for "
	     "dispatch_%ld: %s"),
	   dispatch_id.handle, get_status_string (status));

  amd_dbgapi_agent_id_t agent_id;
  status = amd_dbgapi_dispatch_get_info (dispatch_id,
					 AMD_DBGAPI_DISPATCH_INFO_AGENT,
					 sizeof (agent_id), &agent_id);
  if (status != AMD_DBGAPI_STATUS_SUCCESS)
    error (_("amd_dbgapi_dispatch_get_info failed to get agent id for "
	     "dispatch_%ld: %s"),
	   dispatch_id.handle, get_status_string (status));

  amd_dbgapi_os_queue_packet_id_t os_id;
  status = amd_dbgapi_dispatch_get_info
    (dispatch_id, AMD_DBGAPI_DISPATCH_INFO_OS_QUEUE_PACKET_ID,
     sizeof (os_id), &os_id);
  if (status != AMD_DBGAPI_STATUS_SUCCESS)
    error (_("amd_dbgapi_dispatch_get_info failed to get OS queue packet id "
	     "for dispatch_%ld: %s"),
	   dispatch_id.handle, get_status_string (status));

  return string_printf ("AMDGPU Dispatch %ld:%ld:%ld (PKID %ld)",
			agent_id.handle, queue_id.handle, dispatch_id.handle,
			os_id);
}

/* Return the dispatch position string for a given thread.  */

static std::string
dispatch_pos_string (thread_info *tp)
{
  uint32_t group_ids[3];
  if (wave_get_info (tp, AMD_DBGAPI_WAVE_INFO_WORKGROUP_COORD, group_ids)
      != AMD_DBGAPI_STATUS_SUCCESS)
    return "(?,?,?)/?";

  uint32_t wave_in_group;
  wave_get_info_throw (tp, AMD_DBGAPI_WAVE_INFO_WAVE_NUMBER_IN_WORKGROUP,
		       wave_in_group);

  return string_printf ("(%d,%d,%d)/%d",
			group_ids[0], group_ids[1], group_ids[2],
			wave_in_group);
}

/* Return the target id string for a given queue.  */
static std::string
queue_target_id_string (amd_dbgapi_queue_id_t queue_id)
{
  amd_dbgapi_agent_id_t agent_id;
  amd_dbgapi_status_t status
    = amd_dbgapi_queue_get_info (queue_id, AMD_DBGAPI_QUEUE_INFO_AGENT,
				 sizeof (agent_id), &agent_id);
  if (status != AMD_DBGAPI_STATUS_SUCCESS)
    error (_("amd_dbgapi_queue_get_info failed to get agent id for "
	     "queue_%ld: %s"),
	   queue_id.handle, get_status_string (status));

  amd_dbgapi_os_queue_id_t os_id;
  status = amd_dbgapi_queue_get_info (queue_id, AMD_DBGAPI_QUEUE_INFO_OS_ID,
				      sizeof (os_id), &os_id);
  if (status != AMD_DBGAPI_STATUS_SUCCESS)
    error (_("amd_dbgapi_queue_get_info failed to get OS queue id for "
	     "queue_%ld: %s"),
	   queue_id.handle, get_status_string (status));

  return string_printf ("AMDGPU Queue %ld:%ld (QID %ld)", agent_id.handle,
			queue_id.handle, os_id);
}

/* Return the target id string for a given agent.  */
static std::string
agent_target_id_string (amd_dbgapi_agent_id_t agent_id)
{
  amd_dbgapi_os_agent_id_t os_id;
  amd_dbgapi_status_t status
    = amd_dbgapi_agent_get_info (agent_id, AMD_DBGAPI_AGENT_INFO_OS_ID,
				 sizeof (os_id), &os_id);
  if (status != AMD_DBGAPI_STATUS_SUCCESS)
    error (_("amd_dbgapi_agent_get_info failed to get OS agent id for "
	     "agent_%ld: %s"),
	   agent_id.handle, get_status_string (status));

  return string_printf ("AMDGPU Agent (GPUID %ld)", os_id);
}

/* Return the thread/wave's workgroup position as a string.  */

static std::string
thread_workgroup_pos_string (thread_info *tp)
{
  uint32_t wave_in_group;
  if (wave_get_info (tp, AMD_DBGAPI_WAVE_INFO_WAVE_NUMBER_IN_WORKGROUP,
		     wave_in_group) != AMD_DBGAPI_STATUS_SUCCESS)
    return "?";

  return string_printf ("%u", wave_in_group);
}

/* Convert flat ID FLATID to coordinates and store them in COORD_ID.
   SIZES is the sizes of each axis.  */

static void
flatid_to_id (uint32_t coord_id[3], size_t flatid, const size_t sizes[3])
{
  coord_id[2] = flatid / (sizes[0] * sizes[1]);

  flatid -= (size_t) coord_id[2] * sizes[0] * sizes[1];

  coord_id[1] = flatid / sizes[0];

  flatid -= (size_t) coord_id[1] * sizes[0];

  coord_id[0] = flatid;
}

/* Object used to collect information about a work-item.  Used to
   compute work-item coordinates taking into account partial
   work-groups.  */

struct work_item_info
{
  amd_dbgapi_dispatch_id_t dispatch_id;
  amd_dbgapi_queue_id_t queue_id;
  amd_dbgapi_agent_id_t agent_id;

  /* Grid sizes in work-items.  */
  uint32_t grid_sizes[3];

  /* Grid's work-group sizes in work-items.  */
  uint16_t work_group_sizes[3];

  /* Grid work-group coordinates.  */
  uint32_t work_group_ids[3];

  /* Wave in work-group.  */
  uint32_t wave_in_group;

  /* Lane count per wave.  */
  size_t lane_count;

  /* Return the flat work-item id of the lane at index LANE_INDEX.  */
  size_t flatid (int lane_index) const
  {
    return wave_in_group * lane_count + lane_index;
  }

  /* Store in PARTIAL_WORKGROUP_SIZES the work-group item sizes for
     each axis, taking into account the work-items that actually fit
     in the grid.  */
  void partial_work_group_sizes (size_t partial_work_group_sizes[3]) const
  {
    for (int i = 0; i < 3; i++)
      {
	size_t work_item_start = work_group_ids[i] * work_group_sizes[i];
	size_t work_item_end = work_item_start + work_group_sizes[i];
	if (work_item_end > grid_sizes[i])
	  work_item_end = grid_sizes[i];
	partial_work_group_sizes[i] = work_item_end - work_item_start;
      }
  }
};

/* Populate WI, a work_item_info object describing lane LANE of wave
   TP.  Returns true on success, false if info is not available.  */

static bool
make_work_item_info (thread_info *tp, int lane, work_item_info *wi)
{
  if (wave_get_info (tp, AMD_DBGAPI_WAVE_INFO_DISPATCH, wi->dispatch_id)
      != AMD_DBGAPI_STATUS_SUCCESS)
    {
      /* The dispatch associated with a wave is not available.  A wave
	 may not have an associated dispatch if attaching to a process
	 with already existing waves.  */
      return false;
    }

  wave_get_info_throw (tp, AMD_DBGAPI_WAVE_INFO_QUEUE, wi->queue_id);
  wave_get_info_throw (tp, AMD_DBGAPI_WAVE_INFO_AGENT, wi->agent_id);
  dispatch_get_info_throw (wi->dispatch_id, AMD_DBGAPI_DISPATCH_INFO_GRID_SIZES,
			   wi->grid_sizes);

  dispatch_get_info_throw (wi->dispatch_id,
			   AMD_DBGAPI_DISPATCH_INFO_WORKGROUP_SIZES,
			   wi->work_group_sizes);

  wave_get_info_throw (tp, AMD_DBGAPI_WAVE_INFO_WORKGROUP_COORD,
		       wi->work_group_ids);

  wave_get_info_throw (tp, AMD_DBGAPI_WAVE_INFO_WAVE_NUMBER_IN_WORKGROUP,
		       wi->wave_in_group);

  wave_get_info_throw (tp, AMD_DBGAPI_WAVE_INFO_LANE_COUNT, wi->lane_count);

  return true;
}

/* Return the lane's work-group position as a string.  */

static std::string
lane_workgroup_pos_string (thread_info *tp, int lane)
{
  work_item_info wi;

  if (make_work_item_info (tp, lane, &wi))
    {
      size_t partial_work_group_sizes[3];

      wi.partial_work_group_sizes (partial_work_group_sizes);

      size_t work_item_flatid = wi.flatid (lane);

      uint32_t work_item_ids[3];
      flatid_to_id (work_item_ids, work_item_flatid, partial_work_group_sizes);

      return string_printf ("[%d,%d,%d]",
			    work_item_ids[0], work_item_ids[1], work_item_ids[2]);
    }
  else
    return "[?,?,?]";
}

/* Return the target id string for a given lane.  */

static std::string
lane_target_id_string (thread_info *tp, int lane)
{
  amd_dbgapi_dispatch_id_t dispatch_id;
  amd_dbgapi_queue_id_t queue_id;
  amd_dbgapi_agent_id_t agent_id;
  uint32_t group_ids[3];

  std::string str = "AMDGPU Lane ";

  str += (wave_get_info (tp, AMD_DBGAPI_WAVE_INFO_AGENT, agent_id)
	  == AMD_DBGAPI_STATUS_SUCCESS)
	   ? string_printf ("%ld", agent_id.handle)
	   : " ?";

  str += (wave_get_info (tp, AMD_DBGAPI_WAVE_INFO_QUEUE, queue_id)
	  == AMD_DBGAPI_STATUS_SUCCESS)
	   ? string_printf (":%ld", queue_id.handle)
	   : ":?";

  str += (wave_get_info (tp, AMD_DBGAPI_WAVE_INFO_DISPATCH,
			 dispatch_id)
	  == AMD_DBGAPI_STATUS_SUCCESS)
	   ? string_printf (":%ld", dispatch_id.handle)
	   : ":?";

  amd_dbgapi_wave_id_t wave_id = get_amd_dbgapi_wave_id (tp->ptid);

  str += string_printf (":%ld/%d", wave_id.handle, lane);

  str += (wave_get_info (tp, AMD_DBGAPI_WAVE_INFO_WORKGROUP_COORD,
			 group_ids)
	  == AMD_DBGAPI_STATUS_SUCCESS
	  ? string_printf (" (%d,%d,%d)", group_ids[0], group_ids[1],
			   group_ids[2])
	  : " (?,?,?)");

  str += lane_workgroup_pos_string (tp, lane);

  return str;
}

static void
async_event_handler_clear ()
{
  gdb_assert (amd_dbgapi_async_event_handler != nullptr);
  clear_async_event_handler (amd_dbgapi_async_event_handler);
}

static void
async_event_handler_mark ()
{
  gdb_assert (amd_dbgapi_async_event_handler != nullptr);
  mark_async_event_handler (amd_dbgapi_async_event_handler);
}

/* Fetch the amd_dbgapi_inferior_info data for the given inferior.  */

static struct amd_dbgapi_inferior_info *
get_amd_dbgapi_inferior_info (struct inferior *inferior)
{
  if (!inferior)
    inferior = current_inferior ();

  amd_dbgapi_inferior_info *info = amd_dbgapi_inferior_data.get (inferior);

  if (!info)
    info = amd_dbgapi_inferior_data.emplace (inferior, inferior);

  return info;
}

/* Set forward progress requirement to REQUIRE for all processes matching
   PTID.  */

static void
require_forward_progress (ptid_t ptid, process_stratum_target *proc_target,
			  bool require)
{
  for (inferior *inf : all_inferiors (proc_target))
    {
      if (ptid != minus_one_ptid && inf->pid != ptid.pid ())
	continue;

      amd_dbgapi_inferior_info *info = get_amd_dbgapi_inferior_info (inf);

      if (info->process_id == AMD_DBGAPI_PROCESS_NONE)
	continue;

      /* Don't to unnecessary calls to dbgapi to avoid polluting the logs.  */
      if (info->forward_progress_required == require)
	continue;

      amd_dbgapi_status_t status
	= amd_dbgapi_process_set_progress
	    (info->process_id, (require
				? AMD_DBGAPI_PROGRESS_NORMAL
				: AMD_DBGAPI_PROGRESS_NO_FORWARD));
      gdb_assert (status == AMD_DBGAPI_STATUS_SUCCESS);

      info->forward_progress_required = require;

      /* If ptid targets a single inferior and we have found it, no need to
         continue.  */
      if (ptid != minus_one_ptid)
	break;
    }
}

/* Fetch the amd_dbgapi_process_id for the given inferior.  */

amd_dbgapi_process_id_t
get_amd_dbgapi_process_id (struct inferior *inferior)
{
  return get_amd_dbgapi_inferior_info (inferior)->process_id;
}

static void
amd_dbgapi_target_breakpoint_re_set (struct breakpoint *b)
{
}

static void
amd_dbgapi_target_breakpoint_check_status (struct bpstat *bs)
{
  struct amd_dbgapi_inferior_info *info = get_amd_dbgapi_inferior_info ();
  amd_dbgapi_status_t status;

  bs->stop = 0;
  bs->print_it = print_it_noop;

  /* Find the address the breakpoint is set at.  */
  auto it
    = std::find_if (info->breakpoint_map.begin (), info->breakpoint_map.end (),
		    [=] (
		      const decltype (info->breakpoint_map)::value_type &value)
		    { return value.second == bs->breakpoint_at; });

  if (it == info->breakpoint_map.end ())
    error (_ ("Could not find breakpoint_id for breakpoint at %#lx"),
	   bs->bp_location_at->address);

  amd_dbgapi_breakpoint_id_t breakpoint_id{ it->first };
  amd_dbgapi_breakpoint_action_t action;

  status = amd_dbgapi_report_breakpoint_hit (breakpoint_id,
					     reinterpret_cast<
					       amd_dbgapi_client_thread_id_t> (
					       inferior_thread ()),
					     &action);

  if (status != AMD_DBGAPI_STATUS_SUCCESS)
    error (_("amd_dbgapi_report_breakpoint_hit failed: breakpoint_%ld "
	     "at %#lx (%s)"),
	   breakpoint_id.handle, bs->bp_location_at->address,
	   get_status_string (status));

  if (action == AMD_DBGAPI_BREAKPOINT_ACTION_RESUME)
    return;

  /* If the action is AMD_DBGAPI_BREAKPOINT_ACTION_HALT, we need to wait until
     a breakpoint resume event for this breakpoint_id is seen.  */

  amd_dbgapi_event_id_t resume_event_id
    = process_event_queue (info->process_id,
			   AMD_DBGAPI_EVENT_KIND_BREAKPOINT_RESUME);

  /* We should always get a breakpoint_resume event after processing all
     events generated by reporting the breakpoint was hit.  */
  gdb_assert (resume_event_id != AMD_DBGAPI_EVENT_NONE);

  amd_dbgapi_breakpoint_id_t resume_breakpoint_id;
  status = amd_dbgapi_event_get_info (resume_event_id,
				      AMD_DBGAPI_EVENT_INFO_BREAKPOINT,
				      sizeof (resume_breakpoint_id),
				      &resume_breakpoint_id);

  if (status != AMD_DBGAPI_STATUS_SUCCESS)
    error (_("amd_dbgapi_event_get_info failed (%s)"), get_status_string (status));

  /* The debugger API guarantees that [breakpoint_hit...resume_breakpoint]
     sequences cannot interleave, so this breakpoint resume event must be
     for our breakpoint_id.  */
  if (resume_breakpoint_id != breakpoint_id)
    error (_ ("breakpoint resume event is not for this breakpoint. "
	      "Expected breakpoint_%ld, got breakpoint_%ld"),
	   breakpoint_id.handle, resume_breakpoint_id.handle);

  amd_dbgapi_event_processed (resume_event_id);
}

bool
amd_dbgapi_target::thread_alive (ptid_t ptid)
{
  if (!ptid_is_gpu (ptid))
    return beneath ()->thread_alive (ptid);

  /* Check that the wave_id is valid.  */

  amd_dbgapi_wave_state_t state;
  return amd_dbgapi_wave_get_info (get_amd_dbgapi_wave_id (ptid),
				   AMD_DBGAPI_WAVE_INFO_STATE, sizeof (state),
				   &state)
	 == AMD_DBGAPI_STATUS_SUCCESS;
}

const char *
amd_dbgapi_target::thread_name (thread_info *tp)
{
  if (!ptid_is_gpu (tp->ptid))
    return beneath ()->thread_name (tp);

  /* Return the process's comm value—that is, the command name associated with
     the process.  */

  char comm_path[128];
  xsnprintf (comm_path, sizeof (comm_path), "/proc/%ld/comm",
	     (long) tp->ptid.pid ());

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
amd_dbgapi_target::pid_to_str (ptid_t ptid)
{
  if (!ptid_is_gpu (ptid))
    {
      return beneath ()->pid_to_str (ptid);
    }

  return wave_target_id_string (get_amd_dbgapi_wave_id (ptid));
}

std::string
amd_dbgapi_target::lane_to_str (thread_info *thr, int lane)
{
  if (!ptid_is_gpu (thr->ptid))
    return beneath ()->lane_to_str (thr, lane);

  return lane_target_id_string (thr, lane);
}

/* Implementation of target_workgroup_pos_str.  */

std::string
amd_dbgapi_target::dispatch_pos_str (thread_info *thr)
{
  if (!ptid_is_gpu (thr->ptid))
    return beneath ()->dispatch_pos_str (thr);

  return dispatch_pos_string (thr);
}

std::string
amd_dbgapi_target::thread_workgroup_pos_str (thread_info *thr)
{
  if (!ptid_is_gpu (thr->ptid))
    return beneath ()->thread_workgroup_pos_str (thr);

  return thread_workgroup_pos_string (thr);
}

/* Implementation of target_lane_workgroup_pos_str.  */

std::string
amd_dbgapi_target::lane_workgroup_pos_str (thread_info *thr, int lane)
{
  if (!ptid_is_gpu (thr->ptid))
    return beneath ()->lane_workgroup_pos_str (thr, lane);

  return lane_workgroup_pos_string (thr, lane);
}

const char *
amd_dbgapi_target::extra_thread_info (thread_info *tp)
{
  if (!ptid_is_gpu (tp->ptid))
    beneath ()->extra_thread_info (tp);

  return NULL;
}

enum target_xfer_status
amd_dbgapi_target::xfer_partial (enum target_object object, const char *annex,
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
	= (uint64_t) amdgpu_address_space_id_from_core_address (offset);

      amd_dbgapi_segment_address_t segment_address
	= amdgpu_segment_address_from_core_address (offset);

      amd_dbgapi_process_id_t process_id = get_amd_dbgapi_process_id ();
      amd_dbgapi_wave_id_t wave_id = get_amd_dbgapi_wave_id (inferior_ptid);

      amd_dbgapi_architecture_id_t architecture_id;
      amd_dbgapi_address_space_id_t address_space_id;

      if (amd_dbgapi_wave_get_info (wave_id, AMD_DBGAPI_WAVE_INFO_ARCHITECTURE,
				    sizeof (architecture_id), &architecture_id)
	    != AMD_DBGAPI_STATUS_SUCCESS
	  || amd_dbgapi_dwarf_address_space_to_address_space (
	       architecture_id, dwarf_address_space, &address_space_id)
	       != AMD_DBGAPI_STATUS_SUCCESS)
	return TARGET_XFER_EOF;

      size_t len = requested_len;
      amd_dbgapi_status_t status;
      int current_lane = inferior_thread ()->current_simd_lane ();

      if (readbuf)
	status
	  = amd_dbgapi_read_memory (process_id, wave_id, current_lane,
				    address_space_id, segment_address,
				    &len, readbuf);
      else
	status
	  = amd_dbgapi_write_memory (process_id, wave_id, current_lane,
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

static int
insert_one_watchpoint (amd_dbgapi_inferior_info *info, CORE_ADDR addr, int len)
{
  amd_dbgapi_watchpoint_id_t watch_id;
  amd_dbgapi_global_address_t adjusted_address;
  amd_dbgapi_size_t adjusted_size;

  if (amd_dbgapi_set_watchpoint (info->process_id, addr, len,
				 AMD_DBGAPI_WATCHPOINT_KIND_STORE_AND_RMW,
				 &watch_id)
      != AMD_DBGAPI_STATUS_SUCCESS)
    return 1;

  auto cleanup = make_scope_exit ([&] ()
    { amd_dbgapi_remove_watchpoint (watch_id); });

  /* FIXME: A reduced range watchpoint may have been inserted, which would
     require additional watchpoints to be inserted to cover the requested
     range.  */

  if (amd_dbgapi_watchpoint_get_info (watch_id,
				      AMD_DBGAPI_WATCHPOINT_INFO_ADDRESS,
				      sizeof (adjusted_address),
				      &adjusted_address)
	!= AMD_DBGAPI_STATUS_SUCCESS
      || adjusted_address > addr)
    return 1;

  if (amd_dbgapi_watchpoint_get_info (watch_id,
				      AMD_DBGAPI_WATCHPOINT_INFO_SIZE,
				      sizeof (adjusted_size), &adjusted_size)
	!= AMD_DBGAPI_STATUS_SUCCESS
      || (adjusted_address + adjusted_size) < (addr + len))
    return 1;

  if (!info->watchpoint_map
	 .emplace (addr, std::make_pair (addr + len, watch_id))
	 .second)
    return 1;

  cleanup.release ();
  return 0;
}

static void
insert_initial_watchpoints (amd_dbgapi_inferior_info *info)
{
  gdb_assert (info->runtime_state == AMD_DBGAPI_RUNTIME_STATE_LOADED_SUCCESS);

  for (bp_location *loc : all_bp_locations ())
    {
      if (loc->loc_type == bp_loc_hardware_watchpoint
	  && loc->pspace == info->inf->pspace)
	{
	  if (insert_one_watchpoint (info, loc->address, loc->length) != 0)
	    warning (_ (
	      "Failed to insert existing watchpoint after loading runtime."));
	}
    };
}

int
amd_dbgapi_target::insert_watchpoint (CORE_ADDR addr, int len,
				    enum target_hw_bp_type type,
				    struct expression *cond)
{
  struct amd_dbgapi_inferior_info *info = get_amd_dbgapi_inferior_info ();

  if (info->runtime_state == AMD_DBGAPI_RUNTIME_STATE_LOADED_SUCCESS
      && type != hw_write)
    /* We only allow write watchpoints when GPU debugging is active.  */
    return 1;

  int ret = beneath ()->insert_watchpoint (addr, len, type, cond);
  if (ret || info->runtime_state != AMD_DBGAPI_RUNTIME_STATE_LOADED_SUCCESS)
    return ret;

  ret = insert_one_watchpoint (info, addr, len);
  if (ret != 0)
    {
      /* We failed to insert the GPU watchpoint, so remove the CPU watchpoint
	 before returning an error.  */
      beneath ()->remove_watchpoint (addr, len, type, cond);
    }

  return ret;
}

int
amd_dbgapi_target::remove_watchpoint (CORE_ADDR addr, int len,
				    enum target_hw_bp_type type,
				    struct expression *cond)
{
  struct amd_dbgapi_inferior_info *info = get_amd_dbgapi_inferior_info ();

  int ret = beneath ()->remove_watchpoint (addr, len, type, cond);
  if (info->runtime_state != AMD_DBGAPI_RUNTIME_STATE_LOADED_SUCCESS)
    return ret;

  /* Find the watch_id for the addr..addr+len range.  */
  auto it = info->watchpoint_map.upper_bound (addr);
  if (it == info->watchpoint_map.begin ())
    return 1;

  std::advance (it, -1);
  if (addr < it->first || (addr + len) > it->second.first)
    return 1;

  gdb_assert (type == hw_write);

  amd_dbgapi_watchpoint_id_t watch_id = it->second.second;
  info->watchpoint_map.erase (it);
  if (amd_dbgapi_remove_watchpoint (watch_id)
      != AMD_DBGAPI_STATUS_SUCCESS)
    return 1;

  return ret;
}

bool
amd_dbgapi_target::stopped_by_watchpoint ()
{
  if (!ptid_is_gpu (inferior_ptid))
    return beneath ()->stopped_by_watchpoint ();

  amd_dbgapi_watchpoint_list_t watchpoints;
  if (amd_dbgapi_wave_get_info (get_amd_dbgapi_wave_id (inferior_ptid),
				AMD_DBGAPI_WAVE_INFO_WATCHPOINTS,
				sizeof (watchpoints), &watchpoints)
      != AMD_DBGAPI_STATUS_SUCCESS)
    return false;

  free (watchpoints.watchpoint_ids);
  return watchpoints.count != 0;
}

bool
amd_dbgapi_target::stopped_data_address (CORE_ADDR *addr_p)
{
  struct amd_dbgapi_inferior_info *info = get_amd_dbgapi_inferior_info ();

  if (!ptid_is_gpu (inferior_ptid))
    return beneath ()->stopped_data_address (addr_p);

  amd_dbgapi_watchpoint_list_t watchpoints;
  if (amd_dbgapi_wave_get_info (get_amd_dbgapi_wave_id (inferior_ptid),
				AMD_DBGAPI_WAVE_INFO_WATCHPOINTS,
				sizeof (watchpoints), &watchpoints)
      != AMD_DBGAPI_STATUS_SUCCESS)
    return false;

  /* Compute the intersection between the triggered watchpoint ranges.  */
  CORE_ADDR start = std::numeric_limits<CORE_ADDR>::min ();
  CORE_ADDR finish = std::numeric_limits<CORE_ADDR>::max ();
  for (size_t i = 0; i < watchpoints.count; ++i)
    {
      amd_dbgapi_watchpoint_id_t watchpoint = watchpoints.watchpoint_ids[i];
      auto it
	= std::find_if (info->watchpoint_map.begin (),
			info->watchpoint_map.end (),
			[watchpoint] (
			  const decltype (info->watchpoint_map)::value_type
			    &value)
			{ return value.second.second == watchpoint; });
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
amd_dbgapi_target::resume (ptid_t ptid, int step, enum gdb_signal signo)
{
  gdb_assert (!current_inferior ()->process_target ()->commit_resumed_state);

  if (debug_infrun)
    fprintf_unfiltered (gdb_stdlog,
			"\e[1;34minfrun: amd_dbgapi_target::resume "
			"([%d,%ld,%ld])\e[0m\n",
			ptid.pid (), ptid.lwp (), ptid.tid ());

  bool many_threads = ptid == minus_one_ptid || ptid.is_pid ();

  /* The amd_dbgapi_exceptions_t matching signo will only be used if the
     thread which is the target of the signal SIGNO is a GPU thread.  If so,
     make sure that there is a corresponding amd_dbgapi_exceptions_t for SIGNO
     before we try to resume any thread.

     The target thread is:
     - INFERIOR_PTID if ptid is a wildcard pid
     - PTID otherwise.  */
  amd_dbgapi_exceptions_t exception = AMD_DBGAPI_EXCEPTION_NONE;
  if ((many_threads && ptid_is_gpu (inferior_ptid)) || ptid_is_gpu (ptid))
    {
      switch (signo)
	{
	case GDB_SIGNAL_BUS:
	  exception = AMD_DBGAPI_EXCEPTION_WAVE_APERTURE_VIOLATION;
	  break;
	case GDB_SIGNAL_SEGV:
	  exception = AMD_DBGAPI_EXCEPTION_WAVE_MEMORY_VIOLATION;
	  break;
	case GDB_SIGNAL_ILL:
	  exception = AMD_DBGAPI_EXCEPTION_WAVE_ILLEGAL_INSTRUCTION;
	  break;
	case GDB_SIGNAL_FPE:
	  exception = AMD_DBGAPI_EXCEPTION_WAVE_MATH_ERROR;
	  break;
	case GDB_SIGNAL_ABRT:
	  exception = AMD_DBGAPI_EXCEPTION_WAVE_ABORT;
	  break;
	case GDB_SIGNAL_TRAP:
	  exception = AMD_DBGAPI_EXCEPTION_WAVE_TRAP;
	  break;
	case GDB_SIGNAL_0:
	  exception = AMD_DBGAPI_EXCEPTION_NONE;
	  break;
	default:
	  error (_ ("Resuming with signal %s is not supported by this agent."),
		 gdb_signal_to_name (signo));
	}
    }

  if (!ptid_is_gpu (ptid) || many_threads)
    {
      beneath ()->resume (ptid, step, signo);

      /* The request is for a single thread, we are done.  */
      if (!many_threads)
	return;
    }

  process_stratum_target *proc_target = current_inferior ()->process_target ();

  /* Disable forward progress requirement.  */
  require_forward_progress (ptid, proc_target, false);

  for (thread_info *thread :
       all_non_exited_threads (current_inferior ()->process_target (), ptid))
    {
      if (!ptid_is_gpu (thread->ptid))
	continue;

      amd_dbgapi_wave_id_t wave_id = get_amd_dbgapi_wave_id (thread->ptid);
      amd_dbgapi_status_t status;
      if (thread->ptid == inferior_ptid)
	status
	  = amd_dbgapi_wave_resume (wave_id,
				    (step
				     ? AMD_DBGAPI_RESUME_MODE_SINGLE_STEP
				     : AMD_DBGAPI_RESUME_MODE_NORMAL),
				    exception);
      else
	status
	  = amd_dbgapi_wave_resume (wave_id, AMD_DBGAPI_RESUME_MODE_NORMAL,
				    AMD_DBGAPI_EXCEPTION_NONE);

      if (status != AMD_DBGAPI_STATUS_SUCCESS
	  /* Ignore the error that wave is no longer valid as that could
             indicate that the process has exited.  GDB treats resuming a
	     thread that no longer exists as being successful.  */
	  && status != AMD_DBGAPI_STATUS_ERROR_INVALID_WAVE_ID)
	error (_("wave_resume for wave_%ld failed (%s)"), wave_id.handle,
	       get_status_string (status));
    }
}

void
amd_dbgapi_target::commit_resumed ()
{
  if (debug_infrun)
    fprintf_unfiltered (gdb_stdlog,
			"\e[1;34minfrun: amd_dbgapi_target::commit_resumed "
			"()\e[0m\n");

  beneath ()->commit_resumed ();

  process_stratum_target *proc_target = current_inferior ()->process_target ();
  require_forward_progress (minus_one_ptid, proc_target, true);
}

void
amd_dbgapi_target::stop (ptid_t ptid)
{
  gdb_assert (!current_inferior ()->process_target ()->commit_resumed_state);

  if (debug_infrun)
    fprintf_unfiltered (gdb_stdlog,
			"\e[1;34minfrun: amd_dbgapi_target::stop "
			"([%d,%ld,%ld])\e[0m\n",
			ptid.pid (), ptid.lwp (), ptid.tid ());

  bool many_threads = ptid == minus_one_ptid || ptid.is_pid ();

  if (!ptid_is_gpu (ptid) || many_threads)
    {
      beneath ()->stop (ptid);

      /* The request is for a single thread, we are done.  */
      if (!many_threads)
	return;
    }

  auto stop_one_thread = [this] (thread_info *thread)
  {
    gdb_assert (thread != nullptr);

    amd_dbgapi_wave_id_t wave_id = get_amd_dbgapi_wave_id (thread->ptid);
    amd_dbgapi_wave_state_t state;
    amd_dbgapi_status_t status
      = amd_dbgapi_wave_get_info (wave_id, AMD_DBGAPI_WAVE_INFO_STATE,
				  sizeof (state), &state);
    if (status == AMD_DBGAPI_STATUS_SUCCESS)
      {
	/* If the wave is already known to be stopped then do nothing.  */
	if (state == AMD_DBGAPI_WAVE_STATE_STOP)
	  return;

	status = amd_dbgapi_wave_stop (wave_id);
	if (status == AMD_DBGAPI_STATUS_SUCCESS)
	  return;

	if (status != AMD_DBGAPI_STATUS_ERROR_INVALID_WAVE_ID)
	  error (_("wave_stop for wave_%ld failed (%s)"), wave_id.handle,
		 get_status_string (status));
      }
    else if (status != AMD_DBGAPI_STATUS_ERROR_INVALID_WAVE_ID)
      error (_("wave_get_info for wave_%ld failed (%s)"), wave_id.handle,
	     get_status_string (status));

    /* The status is AMD_DBGAPI_STATUS_ERROR_INVALID_WAVE_ID.  The wave
       could have terminated since the last time the wave list was
       refreshed.  */

    if (report_thread_events)
      {
	target_waitstatus ws;
	ws.set_thread_exited (0);

	get_amd_dbgapi_inferior_info (thread->inf)
	  ->wave_events.emplace_back (thread->ptid, ws);

	if (target_is_async_p ())
	  async_event_handler_mark ();
      }

    delete_thread_silent (thread);
  };

  process_stratum_target *proc_target = current_inferior ()->process_target ();

  /* Disable forward progress requirement.  */
  require_forward_progress (ptid, proc_target, false);

  if (!many_threads)
    {
      /* No need to iterate all non-exited threads if the request is to stop a
	 specific thread.  */
      stop_one_thread (find_thread_ptid (proc_target, ptid));
      return;
    }

  for (auto *inf : all_inferiors (proc_target))
    /* Use the threads_safe iterator since stop_one_thread may delete the
       thread if it has exited.  */
    for (auto *thread : inf->threads_safe ())
      if (thread->state != THREAD_EXITED && thread->ptid.matches (ptid)
	  && ptid_is_gpu (thread->ptid))
	stop_one_thread (thread);
}

static void
handle_target_event (gdb_client_data client_data)
{
  inferior_event_handler (INF_REG_EVENT);
}

/* Called when a dbgapi notifier fd is readable.  CLIENT_DATA is the
   amd_dbgapi_inferior_info object corresponding to the notifier.  */

static void
dbgapi_notifier_handler (int error, gdb_client_data client_data)
{
  amd_dbgapi_inferior_info *info = (amd_dbgapi_inferior_info *) client_data;
  int ret;

  /* Drain the notifier pipe.  */
  do
    {
      char buf;
      ret = read (info->notifier, &buf, 1);
    }
  while (ret >= 0 || (ret == -1 && errno == EINTR));

  /* Signal our async handler.  */
  async_event_handler_mark ();
}

void
amd_dbgapi_target::async (int enable)
{
  infrun_debug_printf ("amd-dbgapi async enable=%d", enable);

  beneath ()->async (enable);

  if (enable)
    {
      if (amd_dbgapi_async_event_handler != nullptr)
	{
	  /* Already enabled.  */
	  return;
	}

      /* The library gives us one notifier file descriptor per inferior (even
	 the ones that have not yet loaded their runtime).  Register them
	 all with the event loop.  */
      process_stratum_target *proc_target
	= current_inferior ()->process_target ();

      for (inferior *inf : all_non_exited_inferiors (proc_target))
	{
	  amd_dbgapi_inferior_info *info = get_amd_dbgapi_inferior_info (inf);

	  if (info->notifier != -1)
	    add_file_handler (info->notifier, dbgapi_notifier_handler, info,
			      "amd-dbgapi dbgapi notifier");
	}

      amd_dbgapi_async_event_handler
	= create_async_event_handler (handle_target_event, nullptr,
				      "amd-dbgapi");

      /* There may be pending events to handle.  Tell the event loop to poll
	 them.  */
      async_event_handler_mark ();
    }
  else
    {
      if (amd_dbgapi_async_event_handler == nullptr)
	return;

      for (inferior *inf : all_inferiors ())
	{
	  amd_dbgapi_inferior_info *info = get_amd_dbgapi_inferior_info (inf);

	  if (info->notifier != -1)
	    delete_file_handler (info->notifier);
	}

      delete_async_event_handler (&amd_dbgapi_async_event_handler);
    }
}

static void
process_one_event (amd_dbgapi_event_id_t event_id,
		   amd_dbgapi_event_kind_t event_kind)
{
  amd_dbgapi_status_t status;

  amd_dbgapi_process_id_t process_id;
  status = amd_dbgapi_event_get_info (event_id, AMD_DBGAPI_EVENT_INFO_PROCESS,
				      sizeof (process_id), &process_id);
  if (status != AMD_DBGAPI_STATUS_SUCCESS)
    error (_("event_get_info for event_%ld failed (%s)"), event_id.handle,
	   get_status_string (status));

  amd_dbgapi_os_process_id_t pid;
  status
    = amd_dbgapi_process_get_info (process_id, AMD_DBGAPI_PROCESS_INFO_OS_ID,
				   sizeof (pid), &pid);
  if (status != AMD_DBGAPI_STATUS_SUCCESS)
    error (_("process_get_info for process_%ld failed (%s)"),
	   process_id.handle, get_status_string (status));

  auto *proc_target = current_inferior ()->process_target ();
  inferior *inf = find_inferior_pid (proc_target, pid);
  gdb_assert (inf != nullptr && "Could not find inferior");
  struct amd_dbgapi_inferior_info *info = get_amd_dbgapi_inferior_info (inf);

  switch (event_kind)
    {
    case AMD_DBGAPI_EVENT_KIND_WAVE_COMMAND_TERMINATED:
    case AMD_DBGAPI_EVENT_KIND_WAVE_STOP:
      {
	amd_dbgapi_wave_id_t wave_id;
	status
	  = amd_dbgapi_event_get_info (event_id, AMD_DBGAPI_EVENT_INFO_WAVE,
				       sizeof (wave_id), &wave_id);
	if (status != AMD_DBGAPI_STATUS_SUCCESS)
	  error (_("event_get_info for event_%ld failed (%s)"),
		 event_id.handle, get_status_string (status));

	ptid_t event_ptid (pid, 1, wave_id.handle);
	target_waitstatus ws;

	amd_dbgapi_wave_stop_reasons_t stop_reason;
	status = amd_dbgapi_wave_get_info (wave_id,
					   AMD_DBGAPI_WAVE_INFO_STOP_REASON,
					   sizeof (stop_reason), &stop_reason);
	if (status == AMD_DBGAPI_STATUS_ERROR_INVALID_WAVE_ID
	    && event_kind == AMD_DBGAPI_EVENT_KIND_WAVE_COMMAND_TERMINATED)
	  {
	    ws.set_thread_exited (0);
	  }
	else if (status == AMD_DBGAPI_STATUS_SUCCESS)
	  {
	    if (stop_reason & AMD_DBGAPI_WAVE_STOP_REASON_APERTURE_VIOLATION)
	      ws.set_stopped (GDB_SIGNAL_BUS);
	    else if (stop_reason
		     & AMD_DBGAPI_WAVE_STOP_REASON_MEMORY_VIOLATION)
	      ws.set_stopped (GDB_SIGNAL_SEGV);
	    else if (stop_reason
		     & AMD_DBGAPI_WAVE_STOP_REASON_ILLEGAL_INSTRUCTION)
	      ws.set_stopped (GDB_SIGNAL_ILL);
	    else if (stop_reason
		     & (AMD_DBGAPI_WAVE_STOP_REASON_FP_INPUT_DENORMAL
			| AMD_DBGAPI_WAVE_STOP_REASON_FP_DIVIDE_BY_0
			| AMD_DBGAPI_WAVE_STOP_REASON_FP_OVERFLOW
			| AMD_DBGAPI_WAVE_STOP_REASON_FP_UNDERFLOW
			| AMD_DBGAPI_WAVE_STOP_REASON_FP_INEXACT
			| AMD_DBGAPI_WAVE_STOP_REASON_FP_INVALID_OPERATION
			| AMD_DBGAPI_WAVE_STOP_REASON_INT_DIVIDE_BY_0))
	      ws.set_stopped (GDB_SIGNAL_FPE);
	    else if (stop_reason
		     & (AMD_DBGAPI_WAVE_STOP_REASON_BREAKPOINT
			| AMD_DBGAPI_WAVE_STOP_REASON_WATCHPOINT
			| AMD_DBGAPI_WAVE_STOP_REASON_SINGLE_STEP
			| AMD_DBGAPI_WAVE_STOP_REASON_DEBUG_TRAP
			| AMD_DBGAPI_WAVE_STOP_REASON_TRAP))
	      ws.set_stopped (GDB_SIGNAL_TRAP);
	    else if (stop_reason & AMD_DBGAPI_WAVE_STOP_REASON_ASSERT_TRAP)
	      ws.set_stopped (GDB_SIGNAL_ABRT);
	    else
	      ws.set_stopped (GDB_SIGNAL_0);

	    thread_info *thread = find_thread_ptid (proc_target, event_ptid);
	    if (thread == nullptr)
	      {
		/* Silently create new GPU threads to avoid spamming the
		   terminal with thousands of "[New Thread ...]" messages.  */
		thread = add_thread_silent (proc_target, event_ptid);
		set_running (proc_target, event_ptid, 1);
		set_executing (proc_target, event_ptid, 1);
	      }

	    /* If the wave is stopped because of a software breakpoint, the
	       program counter needs to be adjusted so that it points to the
	       breakpoint instruction.  */
	    if ((stop_reason & AMD_DBGAPI_WAVE_STOP_REASON_BREAKPOINT) != 0)
	      {
		regcache *regcache = get_thread_regcache (thread);
		gdbarch *gdbarch = regcache->arch ();

		CORE_ADDR pc = regcache_read_pc (regcache);
		CORE_ADDR adjusted_pc
		  = pc - gdbarch_decr_pc_after_break (gdbarch);

		if (adjusted_pc != pc)
		  regcache_write_pc (regcache, adjusted_pc);
	      }
	  }
	else
	  error (_("wave_get_info for wave_%ld failed (%s)"),
		 wave_id.handle, get_status_string (status));

	info->wave_events.emplace_back (event_ptid, ws);
	break;
      }

    case AMD_DBGAPI_EVENT_KIND_CODE_OBJECT_LIST_UPDATED:
      /* We get here when the following sequence of events happens:

	   - the inferior hits the amd-dbgapi "r_brk" internal breakpoint
	   - amd_dbgapi_target_breakpoint::check_status calls
	     amd_dbgapi_report_breakpoint_hit, which queues an event of this
	     kind in dbgapi
	   - amd_dbgapi_target_breakpoint::check_status calls
	     process_event_queue, which pulls the event out of dbgapi, and
	     gets us here

	 When amd_dbgapi_target_breakpoint::check_status is called, the current
	 inferior is the inferior that hit the breakpoint, which should still be
	 the case now.  */
      gdb_assert (inf == current_inferior ());
      handle_solib_event ();
      break;

    case AMD_DBGAPI_EVENT_KIND_BREAKPOINT_RESUME:
      /* Breakpoint resume events should be handled by the breakpoint
	 action, and this code should not reach this.  */
      gdb_assert_not_reached ("unhandled event kind");
      break;

    case AMD_DBGAPI_EVENT_KIND_RUNTIME:
      {
	amd_dbgapi_runtime_state_t runtime_state;

	if ((status
	     = amd_dbgapi_event_get_info (event_id,
					  AMD_DBGAPI_EVENT_INFO_RUNTIME_STATE,
					  sizeof (runtime_state),
					  &runtime_state))
	    != AMD_DBGAPI_STATUS_SUCCESS)
	  error (_("event_get_info for event_%ld failed (%s)"),
		 event_id.handle, get_status_string (status));

	switch (runtime_state)
	  {
	  case AMD_DBGAPI_RUNTIME_STATE_LOADED_SUCCESS:
	    gdb_assert (info->runtime_state
			== AMD_DBGAPI_RUNTIME_STATE_UNLOADED);
	    info->runtime_state = runtime_state;
	    insert_initial_watchpoints (info);
	    break;

	  case AMD_DBGAPI_RUNTIME_STATE_UNLOADED:
	    gdb_assert (info->runtime_state
			!= AMD_DBGAPI_RUNTIME_STATE_UNLOADED);
	    info->runtime_state = runtime_state;
	    break;

	  case AMD_DBGAPI_RUNTIME_STATE_LOADED_ERROR_RESTRICTION:
	    gdb_assert (info->runtime_state
			== AMD_DBGAPI_RUNTIME_STATE_UNLOADED);
	    warning (_("amd-dbgapi: unable to enable GPU debugging "
		       "due to a restriction error"));
	    info->runtime_state = runtime_state;
	    break;
	  }
      }
      break;

    default:
      error (_ ("event kind (%d) not supported"), event_kind);
    }

  amd_dbgapi_event_processed (event_id);
}

/* Drain the dbgapi event queue of a given process_id, or of all processes if
   process_id is AMD_DBGAPI_PROCESS_NONE.  Stop processing the events if an
   event of a given kind is requested and `process_id` is not
   AMD_DBGAPI_PROCESS_NONE. Wave stop events that are not returned are queued
   into their inferior's amd_dbgapi_inferior_info pending wave events. */
static amd_dbgapi_event_id_t
process_event_queue (amd_dbgapi_process_id_t process_id,
		     amd_dbgapi_event_kind_t until_event_kind)
{
  /* An event of a given type can only be requested from a single process_id.
   */
  gdb_assert (until_event_kind == AMD_DBGAPI_EVENT_KIND_NONE
	      || process_id != AMD_DBGAPI_PROCESS_NONE);

  while (true)
    {
      amd_dbgapi_event_id_t event_id;
      amd_dbgapi_event_kind_t event_kind;

      amd_dbgapi_status_t status
	= amd_dbgapi_process_next_pending_event (process_id, &event_id,
						 &event_kind);

      if (status != AMD_DBGAPI_STATUS_SUCCESS)
	error (_("next_pending_event failed (%s)"), get_status_string (status));

      if (event_id == AMD_DBGAPI_EVENT_NONE || event_kind == until_event_kind)
	return event_id;

      process_one_event (event_id, event_kind);
    }
}

bool
amd_dbgapi_target::has_pending_events ()
{
  if (amd_dbgapi_async_event_handler != nullptr
      && async_event_handler_marked (amd_dbgapi_async_event_handler))
    return true;

  return beneath ()->has_pending_events ();
}

static std::pair<ptid_t, target_waitstatus>
consume_one_event (ptid_t ptid)
{
  auto *target = current_inferior ()->process_target ();
  struct amd_dbgapi_inferior_info *info = nullptr;

  if (ptid == minus_one_ptid)
    {
      for (inferior *inf : all_inferiors (target))
	if (!(info = get_amd_dbgapi_inferior_info (inf))->wave_events.empty ())
	  break;
      gdb_assert (info != nullptr);
    }
  else
    {
      gdb_assert (ptid.is_pid ());
      inferior *inf = find_inferior_pid (target, ptid.pid ());

      gdb_assert (inf != nullptr);
      info = get_amd_dbgapi_inferior_info (inf);
    }

  if (info->wave_events.empty ())
    return { minus_one_ptid, {} };

  auto event = info->wave_events.front ();
  info->wave_events.pop_front ();

  return event;
}

ptid_t
amd_dbgapi_target::wait (ptid_t ptid, struct target_waitstatus *ws,
		       target_wait_flags target_options)
{
  gdb_assert (!current_inferior ()->process_target ()->commit_resumed_state);
  gdb_assert (ptid == minus_one_ptid || ptid.is_pid ());

  if (debug_infrun)
    fprintf_unfiltered (gdb_stdlog,
			"\e[1;34minfrun: amd_dbgapi_target::wait (%d, %ld, "
			"%ld)\e[0m\n",
			ptid.pid (), ptid.lwp (), ptid.tid ());

  ptid_t event_ptid = beneath ()->wait (ptid, ws, target_options);
  if (event_ptid != minus_one_ptid)
    {
      if (ws->kind () == TARGET_WAITKIND_EXITED
         || ws->kind () == TARGET_WAITKIND_SIGNALLED)
       {
         /* This inferior has exited so drain its dbgapi event queue.  */
         while (consume_one_event (ptid_t (event_ptid.pid ())).first
                != minus_one_ptid)
           ;
       }
      return event_ptid;
    }

  gdb_assert (ws->kind () == TARGET_WAITKIND_NO_RESUMED
	      || ws->kind () == TARGET_WAITKIND_IGNORE);

  /* Flush the async handler first.  */
  if (target_is_async_p ())
    async_event_handler_clear ();

  /* There may be more events to process (either already in `wave_events` or
     that we need to fetch from dbgapi.  Mark the async event handler so that
     amd_dbgapi_target::wait gets called again and again, until it eventually
     returns minus_one_ptid.  */
  auto more_events = make_scope_exit (
    [] ()
    {
      if (target_is_async_p ())
	async_event_handler_mark ();
    });

  auto *proc_target = current_inferior ()->process_target ();

  /* Disable forward progress for the specified pid in ptid if it isn't
     minus_on_ptid, or all attached processes if ptid is minus_one_ptid.  */
  require_forward_progress (ptid, proc_target, false);

  target_waitstatus gpu_waitstatus;
  std::tie (event_ptid, gpu_waitstatus) = consume_one_event (ptid);
  if (event_ptid == minus_one_ptid)
    {
      /* Drain the events from the amd_dbgapi and preserve the ordering.  */
      process_event_queue ();

      std::tie (event_ptid, gpu_waitstatus) = consume_one_event (ptid);
      if (event_ptid == minus_one_ptid)
	{
	  /* If we requested a specific ptid, and nothing came out, assume
	     another ptid may have more events, otherwise, keep the
	     async_event_handler flushed.  */
	  if (ptid == minus_one_ptid)
	    more_events.release ();

	  if (ws->kind () == TARGET_WAITKIND_NO_RESUMED)
	    {
	      /* We can't easily check that all GPU waves are stopped, and no
		 new waves can be created (the GPU has fixed function hardware
		 to create new threads), so even if the target beneath returns
		 waitkind_no_resumed, we have to report waitkind_ignore if GPU
		 debugging is enabled for at least one resumed inferior handled
		 by the amd-dbgapi target.  */

	      for (inferior *inf : all_inferiors ())
		if (inf->target_at (arch_stratum) == &the_amd_dbgapi_target
		    && get_amd_dbgapi_inferior_info (inf)->runtime_state
			 == AMD_DBGAPI_RUNTIME_STATE_LOADED_SUCCESS)
		  {
		    ws->set_ignore ();
		    break;
		  }
	    }

	  /* There are no events to report, return the target beneath's
	     waitstatus (either IGNORE or NO_RESUMED).  */
	  return minus_one_ptid;
	}
    }

  *ws = gpu_waitstatus;
  return event_ptid;
}

bool
amd_dbgapi_target::stopped_by_sw_breakpoint ()
{
  if (!ptid_is_gpu (inferior_ptid))
    return beneath ()->stopped_by_sw_breakpoint ();

  amd_dbgapi_wave_id_t wave_id = get_amd_dbgapi_wave_id (inferior_ptid);

  amd_dbgapi_wave_stop_reasons_t stop_reason;
  return (amd_dbgapi_wave_get_info (wave_id, AMD_DBGAPI_WAVE_INFO_STOP_REASON,
				    sizeof (stop_reason), &stop_reason)
	  == AMD_DBGAPI_STATUS_SUCCESS)
	 && (stop_reason & AMD_DBGAPI_WAVE_STOP_REASON_BREAKPOINT) != 0;
}

bool
amd_dbgapi_target::stopped_by_hw_breakpoint ()
{
  /* The amd-dbgapi target does not support hw breakpoints.  */
  return !ptid_is_gpu (inferior_ptid)
	 && beneath ()->stopped_by_hw_breakpoint ();
}

/* Set the process's memory access reporting precision.

   The precision can be ::AMD_DBGAPI_MEMORY_PRECISION_PRECISE (waves wait for
   memory instructions to complete before executing further instructions), or
   ::AMD_DBGAPI_MEMORY_PRECISION_NONE (memory instructions execute normally).

   Returns true if the precision is supported by the architecture of all agents
   in the process, or false if at least one agent does not support the
   requested precision.

   An error is thrown if setting the precision results in a status other than
   ::AMD_DBGAPI_STATUS_SUCCESS or ::AMD_DBGAPI_STATUS_ERROR_NOT_SUPPORTED.  */
static bool
set_process_memory_precision (amd_dbgapi_process_id_t process_id,
			      amd_dbgapi_memory_precision_t precision)
{
  amd_dbgapi_status_t status
    = amd_dbgapi_set_memory_precision (process_id, precision);

  if (status == AMD_DBGAPI_STATUS_ERROR_NOT_SUPPORTED)
    return false;
  else if (status != AMD_DBGAPI_STATUS_SUCCESS)
    error (_("amd_dbgapi_set_memory_precision failed (%s)"),
	   get_status_string (status));

  return true;
}

static void
enable_amd_dbgapi (inferior *inf)
{
  if (!target_can_async_p ())
    {
      warning (_("The amd-dbgapi target requires target-async, GPU debugging "
		 "is disabled"));
      return;
    }

  /* dbgapi can't attach to a vfork child (a process born from a vfork that
     hasn't exec'ed yet) while we are still attached to the parent.  It would
     not be useful for us to attach to vfork children anyway, because vfork
     children are very restricted in what they can do (see vfork(2)) and aren't
     going to launch some GPU programs that we need to debug.  To avoid this
     problem, we don't push the amd-dbgapi target / attach dbgapi in vfork
     children.  If a vfork child execs, we'll try enabling the amd-dbgapi target
     through the inferior_execd observer.  */
  if (inf->vfork_parent != nullptr)
    return;

  auto *info = get_amd_dbgapi_inferior_info (inf);

  /* Are we already attached?  */
  if (info->process_id != AMD_DBGAPI_PROCESS_NONE)
    {
      gdb_assert (inf->target_is_pushed (&the_amd_dbgapi_target));
      return;
    }

  amd_dbgapi_status_t status
    = amd_dbgapi_process_attach (reinterpret_cast<
				   amd_dbgapi_client_process_id_t> (inf),
				 &info->process_id);
  if (status == AMD_DBGAPI_STATUS_ERROR_RESTRICTION)
    {
      warning (_("amd-dbgapi: unable to enable GPU debugging due to a "
		 "restriction error"));
      return;
    }
  else if (status != AMD_DBGAPI_STATUS_SUCCESS)
    error (_("Could not attach to process %d (%s)"),
	   inf->pid, get_status_string (status));

  if (amd_dbgapi_process_get_info (info->process_id,
				   AMD_DBGAPI_PROCESS_INFO_NOTIFIER,
				   sizeof (info->notifier), &info->notifier)
      != AMD_DBGAPI_STATUS_SUCCESS)
    {
      amd_dbgapi_process_detach (info->process_id);
      info->process_id = AMD_DBGAPI_PROCESS_NONE;
      error (_ ("Could not retrieve process %d's notifier"), inf->pid);
    }

  amd_dbgapi_memory_precision_t memory_precision
    = info->precise_memory.requested ? AMD_DBGAPI_MEMORY_PRECISION_PRECISE
				     : AMD_DBGAPI_MEMORY_PRECISION_NONE;
  if (set_process_memory_precision (info->process_id, memory_precision))
    info->precise_memory.enabled = info->precise_memory.requested;
  else
    warning (
      _ ("AMDGPU precise memory access reporting could not be enabled."));

  gdb_assert (!inf->target_is_pushed (&the_amd_dbgapi_target));
  inf->push_target (&the_amd_dbgapi_target);

  /* The underlying target will already be async if we are running, but not if
     we are attaching.  */
  if (inf->process_target ()->is_async_p ())
    {
      /* Make sure our async event handler is created.  */
      target_async (1);

      /* If the amd-dbgapi target was already async, it didn't register the new
         fd, so make sure it is registered.  This call is idempotent so it's ok
	 if it's already registered.  */
      add_file_handler (info->notifier, dbgapi_notifier_handler, info,
			"amd-dbgapi notifier");
    }
}

static void
disable_amd_dbgapi (inferior *inf)
{
  auto *info = get_amd_dbgapi_inferior_info (inf);

  if (info->process_id == AMD_DBGAPI_PROCESS_NONE)
    return;

  info->runtime_state = AMD_DBGAPI_RUNTIME_STATE_UNLOADED;

  amd_dbgapi_status_t status = amd_dbgapi_process_detach (info->process_id);
  if (status != AMD_DBGAPI_STATUS_SUCCESS)
    warning (_ ("Could not detach from process %d (%s)"),
	     inf->pid, get_status_string (status));

  gdb_assert (info->notifier != -1);
  delete_file_handler (info->notifier);

  gdb_assert (inf->target_is_pushed (&the_amd_dbgapi_target));
  inf->unpush_target (&the_amd_dbgapi_target);

  /* Delete the breakpoints that are still active.  */
  for (auto &&value : info->breakpoint_map)
    delete_breakpoint (value.second);

  /* Reset the amd_dbgapi_inferior_info, except for precise_memory_mode.  */
  bool precise_memory_requested = info->precise_memory.requested;
  *info = amd_dbgapi_inferior_info (inf);
  info->precise_memory.requested = precise_memory_requested;
}

void
amd_dbgapi_target::mourn_inferior ()
{
  disable_amd_dbgapi (current_inferior ());
  beneath ()->mourn_inferior ();
}

void
amd_dbgapi_target::detach (inferior *inf, int from_tty)
{
  /* We're about to resume the waves by detaching the dbgapi library from the
     inferior, so we need to remove all breakpoints that are still inserted.

     Breakpoints may still be inserted because the inferior may be running in
     non-stop mode, or because GDB changed the default setting to leave all
     breakpoints inserted in all-stop mode when all threads are stopped.
   */
  remove_breakpoints_inf (current_inferior ());

  disable_amd_dbgapi (inf);
  beneath ()->detach (inf, from_tty);
}

void
amd_dbgapi_target::fetch_registers (struct regcache *regcache, int regno)
{
  struct gdbarch *gdbarch = regcache->arch ();

  /* delegate to the host routines when not on the device */

  if (!is_amdgpu_arch (gdbarch))
    {
      beneath ()->fetch_registers (regcache, regno);
      return;
    }

  amdgpu_gdbarch_tdep *tdep = get_amdgpu_gdbarch_tdep (gdbarch);
  amd_dbgapi_wave_id_t wave_id = get_amd_dbgapi_wave_id (regcache->ptid ());
  gdb_byte raw[AMDGPU_MAX_REGISTER_SIZE];

  amd_dbgapi_status_t status
    = amd_dbgapi_read_register (wave_id, tdep->register_ids[regno], 0,
				TYPE_LENGTH (register_type (gdbarch, regno)),
				raw);

  if (status == AMD_DBGAPI_STATUS_SUCCESS)
    {
      regcache->raw_supply (regno, raw);
    }
  else if (status != AMD_DBGAPI_STATUS_ERROR_REGISTER_NOT_AVAILABLE)
    {
      warning (_ ("Couldn't read register %s (#%d)."),
	       gdbarch_register_name (gdbarch, regno), regno);
    }
}

void
amd_dbgapi_target::store_registers (struct regcache *regcache, int regno)
{
  struct gdbarch *gdbarch = regcache->arch ();

  if (!is_amdgpu_arch (gdbarch))
    {
      beneath ()->store_registers (regcache, regno);
      return;
    }

  gdb_byte raw[AMDGPU_MAX_REGISTER_SIZE];
  regcache->raw_collect (regno, &raw);

  amdgpu_gdbarch_tdep *tdep = get_amdgpu_gdbarch_tdep (gdbarch);

  /* If the register has read-only bits, invalidate the value in the regcache
     as the value actualy written may differ.  */
  if (tdep->register_properties[regno]
	& AMD_DBGAPI_REGISTER_PROPERTY_READONLY_BITS)
    regcache->invalidate (regno);

  /* Invalidate all volatile registers if this register has the invalidate
     volatile property.  For example, writting to VCC may change the content
     of STATUS.VCCZ.  */
  if (tdep->register_properties[regno]
        & AMD_DBGAPI_REGISTER_PROPERTY_INVALIDATE_VOLATILE)
    for (size_t r = 0; r < tdep->register_properties.size (); ++r)
      if (tdep->register_properties[r] & AMD_DBGAPI_REGISTER_PROPERTY_VOLATILE)
	regcache->invalidate (r);

  amd_dbgapi_wave_id_t wave_id = get_amd_dbgapi_wave_id (regcache->ptid ());

  amd_dbgapi_status_t status
    = amd_dbgapi_write_register (wave_id, tdep->register_ids[regno], 0,
				 TYPE_LENGTH (register_type (gdbarch, regno)),
				 raw);

  if (status != AMD_DBGAPI_STATUS_SUCCESS)
    {
      warning (_ ("Couldn't write register %s (#%d)."),
	       gdbarch_register_name (gdbarch, regno), regno);
    }
}

/* Fix breakpoints created with an address location while the
   architecture was set to the host (could be fixed in core GDB).  */

static void
amd_dbgapi_target_breakpoint_fixup (struct breakpoint *b)
{
  if (b->location.get ()
      && event_location_type (b->location.get ()) == ADDRESS_LOCATION
      && is_amdgpu_arch (b->loc->gdbarch)
      && !is_amdgpu_arch (b->gdbarch))
    {
      b->gdbarch = b->loc->gdbarch;
    }
}

struct gdbarch *
amd_dbgapi_target::thread_architecture (ptid_t ptid)
{
  static std::result_of<decltype (&ptid_t::tid) (ptid_t)>::type last_tid = 0;
  static struct gdbarch *cached_arch = nullptr;

  if (!ptid_is_gpu (ptid))
    return beneath ()->thread_architecture (ptid);

  /* We can cache the gdbarch for a given wave_id (ptid::tid) because
     wave IDs are unique, and aren't reused.  */
  if (ptid.tid () == last_tid)
    return cached_arch;

  amd_dbgapi_wave_id_t wave_id = get_amd_dbgapi_wave_id (ptid);
  amd_dbgapi_architecture_id_t architecture_id;

  if (amd_dbgapi_wave_get_info (wave_id, AMD_DBGAPI_WAVE_INFO_ARCHITECTURE,
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
  info.bfd_arch_info = bfd_lookup_arch (bfd_arch_amdgcn, elf_amdgpu_machine);
  info.byte_order = BFD_ENDIAN_LITTLE;
  info.osabi = GDB_OSABI_AMDGPU_HSA;

  last_tid = ptid.tid ();
  if (!(cached_arch = gdbarch_find_by_info (info)))
    error (_ ("Couldn't get elf_amdgpu_machine (%#x)"), elf_amdgpu_machine);

  return cached_arch;
}

void
amd_dbgapi_target::update_thread_list ()
{
  for (inferior *inf : all_inferiors ())
    {
      amd_dbgapi_process_id_t process_id;
      amd_dbgapi_wave_id_t *wave_list;
      size_t count;

      process_id = get_amd_dbgapi_process_id (inf);
      if (process_id == AMD_DBGAPI_PROCESS_NONE)
	{
	  /* The inferior may not be attached yet.  */
	  continue;
	}

      amd_dbgapi_changed_t changed;
      amd_dbgapi_status_t status;
      if ((status = amd_dbgapi_process_wave_list (process_id, &count,
						  &wave_list, &changed))
	  != AMD_DBGAPI_STATUS_SUCCESS)
	error (_("amd_dbgapi_wave_list failed (%s)"),
	       get_status_string (status));

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
		  process_id, tid,
		  AMD_DBGAPI_WAVE_INFO_STATE, sizeof (state), &state)
	      != AMD_DBGAPI_STATUS_SUCCESS)
	    continue;*/

	  add_thread_silent (inf->process_target (), wave_ptid);
	  set_running (inf->process_target (), wave_ptid, 1);
	  set_executing (inf->process_target (), wave_ptid, 1);
	}
    }

  /* Give the beneath target a chance to do extra processing.  */
  this->beneath ()->update_thread_list ();
}

displaced_step_prepare_status
amd_dbgapi_target::displaced_step_prepare (thread_info *thread,
					 CORE_ADDR &displaced_pc)
{
  if (!ptid_is_gpu (thread->ptid))
    return beneath ()->displaced_step_prepare (thread, displaced_pc);

  gdb_assert (!thread->displaced_step_state.in_progress ());

  /* Read the bytes that were overwritten by the breakpoint instruction.  */
  CORE_ADDR original_pc = regcache_read_pc (get_thread_regcache (thread));

  gdbarch *arch = get_thread_regcache (thread)->arch ();
  size_t size = get_amdgpu_gdbarch_tdep (arch)->breakpoint_instruction_size;
  gdb::unique_xmalloc_ptr<gdb_byte> overwritten_bytes (
    static_cast<gdb_byte *> (xmalloc (size)));

  /* Read the instruction bytes overwritten by the breakpoint.   */
  int err = target_read_memory (original_pc, overwritten_bytes.get (), size);
  if (err != 0)
    throw_error (MEMORY_ERROR, _ ("Error accessing memory address %s (%s)."),
		 paddress (arch, original_pc), safe_strerror (err));

  amd_dbgapi_wave_id_t wave_id = get_amd_dbgapi_wave_id (thread->ptid);
  amd_dbgapi_displaced_stepping_id_t stepping_id;

  amd_dbgapi_status_t status
    = amd_dbgapi_displaced_stepping_start (wave_id, overwritten_bytes.get (),
					   &stepping_id);

  if (status
      == AMD_DBGAPI_STATUS_ERROR_DISPLACED_STEPPING_BUFFER_NOT_AVAILABLE)
    return DISPLACED_STEP_PREPARE_STATUS_UNAVAILABLE;
  else if (status == AMD_DBGAPI_STATUS_ERROR_ILLEGAL_INSTRUCTION)
    return DISPLACED_STEP_PREPARE_STATUS_CANT;
  else if (status != AMD_DBGAPI_STATUS_SUCCESS)
    error (_("amd_dbgapi_displaced_stepping_start failed (%s)"),
	   get_status_string (status));

  struct amd_dbgapi_inferior_info *info = get_amd_dbgapi_inferior_info (thread->inf);
  if (!info->stepping_id_map.emplace (thread, stepping_id.handle).second)
    {
      amd_dbgapi_displaced_stepping_complete (wave_id, stepping_id);
      error (_ ("Could not insert the displaced stepping id in the map"));
    }

  status = amd_dbgapi_wave_get_info (wave_id, AMD_DBGAPI_WAVE_INFO_PC,
				     sizeof (displaced_pc), &displaced_pc);
  if (status != AMD_DBGAPI_STATUS_SUCCESS)
    {
      amd_dbgapi_displaced_stepping_complete (wave_id, stepping_id);
      error (_("amd_dbgapi_wave_get_info failed (%s)"),
	     get_status_string (status));
    }

  displaced_debug_printf ("selected buffer at %#lx", displaced_pc);

  /* We may have written some registers, so flush the register cache.  */
  registers_changed_thread (thread);

  return DISPLACED_STEP_PREPARE_STATUS_OK;
}

displaced_step_finish_status
amd_dbgapi_target::displaced_step_finish (thread_info *thread, gdb_signal sig)
{
  if (!ptid_is_gpu (thread->ptid))
    return beneath ()->displaced_step_finish (thread, sig);

  gdb_assert (thread->displaced_step_state.in_progress ());

  /* Find the stepping_id for this thread.  */
  struct amd_dbgapi_inferior_info *info = get_amd_dbgapi_inferior_info (thread->inf);
  auto it = info->stepping_id_map.find (thread);
  gdb_assert (it != info->stepping_id_map.end ());

  amd_dbgapi_displaced_stepping_id_t stepping_id{ it->second };
  info->stepping_id_map.erase (it);

  amd_dbgapi_wave_id_t wave_id = get_amd_dbgapi_wave_id (thread->ptid);

  amd_dbgapi_wave_stop_reasons_t stop_reason;
  amd_dbgapi_status_t status
    = amd_dbgapi_wave_get_info (wave_id, AMD_DBGAPI_WAVE_INFO_STOP_REASON,
				sizeof (stop_reason), &stop_reason);

  if (status != AMD_DBGAPI_STATUS_SUCCESS)
    error (_("wave_get_info for wave_%ld failed (%s)"), wave_id.handle,
	   get_status_string (status));

  status = amd_dbgapi_displaced_stepping_complete (wave_id, stepping_id);

  if (status != AMD_DBGAPI_STATUS_SUCCESS)
    error (_("amd_dbgapi_displaced_stepping_complete failed (%s)"),
	   get_status_string (status));

  /* We may have written some registers, so flush the register cache.  */
  registers_changed_thread (thread);

  return (stop_reason & AMD_DBGAPI_WAVE_STOP_REASON_SINGLE_STEP) != 0
	   ? DISPLACED_STEP_FINISH_STATUS_OK
	   : DISPLACED_STEP_FINISH_STATUS_NOT_EXECUTED;
}

static void
amd_dbgapi_target_inferior_created (inferior *inf)
{
  /* If the inferior is not running on the native target (e.g. it is running
     on a remote target), we don't want to deal with it.  */
  if (inf->process_target () != get_native_target ())
    return;

  enable_amd_dbgapi (inf);
}

/* Callback called when an inferior is cloned.  */

static void
amd_dbgapi_target_inferior_cloned (inferior *original_inferior,
			     inferior *new_inferior)
{
  auto *orig_info = get_amd_dbgapi_inferior_info (original_inferior);
  auto *new_info = get_amd_dbgapi_inferior_info (new_inferior);

  /* At this point, the process is not started.  Therefore it is sufficient to
     copy the precise memory request, it will be applied when the process
     starts.  */
  gdb_assert (new_info->process_id == AMD_DBGAPI_PROCESS_NONE);
  new_info->precise_memory.requested = orig_info->precise_memory.requested;
}

void
amd_dbgapi_target::follow_exec (inferior *follow_inf, ptid_t ptid,
			      const char *execd_pathname)
{
  inferior *orig_inf = current_inferior ();

  /* The inferior has EXEC'd and the process image has changed.  The dbgapi is
     attached to the old process image, so we need to detach and re-attach to
     the new process image.  */
  disable_amd_dbgapi (orig_inf);

  beneath ()->follow_exec (follow_inf, ptid, execd_pathname);
  gdb_assert (current_inferior () == follow_inf);

  /* If using "follow-exec-mode new", carry over the precise-memory setting
     to the new inferior (otherwise, FOLLOW_INF and ORIG_INF point to the same
     inferior, so this is a no-op).  */
  get_amd_dbgapi_inferior_info (follow_inf)->precise_memory.requested
    = get_amd_dbgapi_inferior_info (orig_inf)->precise_memory.requested;

  enable_amd_dbgapi (follow_inf);
}

static void
amd_dbgapi_inferior_execd (inferior *inf)
{
  enable_amd_dbgapi (inf);
}

void
amd_dbgapi_target::follow_fork (inferior *child_inf, ptid_t child_ptid,
			      target_waitkind fork_kind, bool follow_child,
			      bool detach_fork)
{
  beneath ()->follow_fork (child_inf, child_ptid, fork_kind, follow_child,
			   detach_fork);

  if (child_inf != nullptr)
    {
      /* Copy precise-memory requested value from parent to child.  */
      amd_dbgapi_inferior_info *parent_info
	= get_amd_dbgapi_inferior_info (current_inferior ());
      amd_dbgapi_inferior_info *child_info = get_amd_dbgapi_inferior_info (child_inf);
      child_info->precise_memory.requested
	= parent_info->precise_memory.requested;

      if (fork_kind != TARGET_WAITKIND_VFORKED)
	{
	  scoped_restore_current_thread restore_thread;
	  switch_to_thread (*child_inf->threads ().begin ());
	  enable_amd_dbgapi (child_inf);
	}
    }
}

static void
amd_dbgapi_target_signal_received (gdb_signal sig)
{
  amd_dbgapi_inferior_info *info = get_amd_dbgapi_inferior_info ();

  if (info->process_id == AMD_DBGAPI_PROCESS_NONE)
    return;

  if (!ptid_is_gpu (inferior_thread ()->ptid))
    return;

  if (sig != GDB_SIGNAL_SEGV && sig != GDB_SIGNAL_BUS)
    return;

  if (!info->precise_memory.enabled)
      printf_filtered ("\
Warning: precise memory violation signal reporting is not enabled, reported\n\
location may not be accurate.  See \"show amdgpu precise-memory\".\n");
}

static void
amd_dbgapi_target_normal_stop (bpstat *bs_list, int print_frame)
{
  if (bs_list == nullptr || !print_frame)
    return;

  amd_dbgapi_inferior_info *info = get_amd_dbgapi_inferior_info ();

  if (info->process_id == AMD_DBGAPI_PROCESS_NONE)
    return;

  if (info->precise_memory.enabled)
    return;

  if (!ptid_is_gpu (inferior_thread ()->ptid))
    return;

  bool found_hardware_watchpoint = false;

  for (bpstat *bs = bs_list; bs != nullptr; bs = bs->next)
    if (bs->breakpoint_at != nullptr
	&& is_hardware_watchpoint(bs->breakpoint_at))
      {
	found_hardware_watchpoint = true;
	break;
      }

  if (!found_hardware_watchpoint)
    return;

  printf_filtered ("\
Warning: precise memory signal reporting is not enabled, watchpoint stop\n\
location may not be accurate.  See \"show amdgpu precise-memory\".\n");
}

static cli_style_option warning_style ("amd_dbgapi_warning", ui_file_style::RED);
static cli_style_option info_style ("amd_dbgapi_info", ui_file_style::GREEN);
static cli_style_option verbose_style ("amd_dbgapi_verbose", ui_file_style::BLUE);

static amd_dbgapi_callbacks_t dbgapi_callbacks = {
  /* allocate_memory.  */
  .allocate_memory = malloc,

  /* deallocate_memory.  */
  .deallocate_memory = free,

  /* get_os_pid.  */
  .get_os_pid = [] (amd_dbgapi_client_process_id_t client_process_id,
		    pid_t *pid) -> amd_dbgapi_status_t
  {
    inferior *inf = reinterpret_cast<inferior *> (client_process_id);

    if (inf->pid == 0)
      return AMD_DBGAPI_STATUS_ERROR_PROCESS_EXITED;

    *pid = inf->pid;
    return AMD_DBGAPI_STATUS_SUCCESS;
  },

  /* set_breakpoint callback.  */
  .insert_breakpoint =
    [] (amd_dbgapi_client_process_id_t client_process_id,
	amd_dbgapi_global_address_t address,
	amd_dbgapi_breakpoint_id_t breakpoint_id)
  {
    inferior *inf = reinterpret_cast<inferior *> (client_process_id);
    struct amd_dbgapi_inferior_info *info = get_amd_dbgapi_inferior_info (inf);

    /* Initialize the breakpoint ops lazily since we depend on
       bkpt_breakpoint_ops and we can't control the order in which
       initializers are called.  */
    if (amd_dbgapi_target_breakpoint_ops.check_status == NULL)
      {
	amd_dbgapi_target_breakpoint_ops = bkpt_breakpoint_ops;
	amd_dbgapi_target_breakpoint_ops.check_status
	  = amd_dbgapi_target_breakpoint_check_status;
	amd_dbgapi_target_breakpoint_ops.re_set
	  = amd_dbgapi_target_breakpoint_re_set;
      }

    auto it = info->breakpoint_map.find (breakpoint_id.handle);
    if (it != info->breakpoint_map.end ())
      return AMD_DBGAPI_STATUS_ERROR_INVALID_BREAKPOINT_ID;

    /* We need to find the address in the given inferior's program space.  */
    scoped_restore_current_thread restore_thread;
    switch_to_inferior_no_thread (inf);

    /* Create a new breakpoint.  */
    struct obj_section *section = find_pc_section (address);
    if (!section || !section->objfile)
      return AMD_DBGAPI_STATUS_ERROR;

    event_location_up location = new_address_location (address, nullptr, 0);
    if (!create_breakpoint (section->objfile->arch (), location.get (),
			    /*cond_string*/ NULL, /*thread*/ -1,
			    /*extra_string*/ NULL, false /*force_condition*/,
			    /*parse_extra*/ 0, /*tempflag*/ 0,
			    /*bptype*/ bp_breakpoint,
			    /*ignore_count*/ 0,
			    /*pending_break*/ AUTO_BOOLEAN_FALSE,
			    /*ops*/ &amd_dbgapi_target_breakpoint_ops, /*from_tty*/ 0,
			    /*enabled*/ 1, /*internal*/ 1, /*flags*/ 0))
      return AMD_DBGAPI_STATUS_ERROR;

    /* Find our breakpoint in the breakpoint list.  */
    breakpoint *bp = nullptr;
    for (breakpoint *b : all_breakpoints ())
      if (b->ops == &amd_dbgapi_target_breakpoint_ops && b->loc
	  && b->loc->pspace->aspace == inf->aspace
	  && b->loc->address == address)
	{
	  bp = b;
	  break;
	}

    if (!bp)
      error (_ ("Could not find breakpoint"));

    info->breakpoint_map.emplace (breakpoint_id.handle, bp);
    return AMD_DBGAPI_STATUS_SUCCESS;
  },

  /* remove_breakpoint callback.  */
  .remove_breakpoint =
    [] (amd_dbgapi_client_process_id_t client_process_id,
	amd_dbgapi_breakpoint_id_t breakpoint_id)
  {
    inferior *inf = reinterpret_cast<inferior *> (client_process_id);
    struct amd_dbgapi_inferior_info *info = get_amd_dbgapi_inferior_info (inf);

    auto it = info->breakpoint_map.find (breakpoint_id.handle);
    if (it == info->breakpoint_map.end ())
      return AMD_DBGAPI_STATUS_ERROR_INVALID_BREAKPOINT_ID;

    delete_breakpoint (it->second);
    info->breakpoint_map.erase (it);

    return AMD_DBGAPI_STATUS_SUCCESS;
  },

  .log_message = [] (amd_dbgapi_log_level_t level, const char *message) -> void
  {
    gdb::optional<target_terminal::scoped_restore_terminal_state> tstate;

    if (level > get_debug_amdgpu_log_level ())
      return;

    if (target_supports_terminal_ours ())
      {
	tstate.emplace ();
	target_terminal::ours_for_output ();
      }

    struct ui_file *out_file
      = (level >= AMD_DBGAPI_LOG_LEVEL_INFO) ? gdb_stdlog : gdb_stderr;

    if (filtered_printing_initialized ())
      out_file->wrap_here (0);

    switch (level)
      {
      case AMD_DBGAPI_LOG_LEVEL_FATAL_ERROR:
	fputs_unfiltered ("amd-dbgapi: ", out_file);
	break;
      case AMD_DBGAPI_LOG_LEVEL_WARNING:
	fputs_styled_unfiltered ("amd-dbgapi: ", warning_style.style (),
				 out_file);
	break;
      case AMD_DBGAPI_LOG_LEVEL_INFO:
	fputs_styled_unfiltered ("amd-dbgapi: ", info_style.style (),
				 out_file);
	break;
      case AMD_DBGAPI_LOG_LEVEL_TRACE:
      case AMD_DBGAPI_LOG_LEVEL_VERBOSE:
	fputs_styled_unfiltered ("amd-dbgapi: ", verbose_style.style (),
				 out_file);
	break;
      }

    fputs_unfiltered (message, out_file);
    fputs_unfiltered ("\n", out_file);
  }
};

void
amd_dbgapi_target::prevent_new_threads (bool prevent, inferior *inf)
{
  beneath ()->prevent_new_threads (prevent, inf);

  amd_dbgapi_inferior_info *info = get_amd_dbgapi_inferior_info ();
  if (info->process_id == AMD_DBGAPI_PROCESS_NONE)
    return;

  amd_dbgapi_wave_creation_t mode
    = (prevent
       ? AMD_DBGAPI_WAVE_CREATION_STOP
       : AMD_DBGAPI_WAVE_CREATION_NORMAL);
  amd_dbgapi_status_t status
    = amd_dbgapi_process_set_wave_creation (info->process_id, mode);
  if (status != AMD_DBGAPI_STATUS_SUCCESS)
    error (_("amd-dbgapi failed to set wave creation mode (%s)"),
	   get_status_string (status));
}

void
amd_dbgapi_target::close ()
{
  /* Finalize and re-initialize the debugger API so that the handle ID numbers
     will all start from the beginning again.  */

  amd_dbgapi_status_t status = amd_dbgapi_finalize ();
  if (status != AMD_DBGAPI_STATUS_SUCCESS)
    error (_("amd-dbgapi failed to finalize (%s)"),
	   get_status_string (status));

  status = amd_dbgapi_initialize (&dbgapi_callbacks);
  if (status != AMD_DBGAPI_STATUS_SUCCESS)
    error (_("amd-dbgapi failed to initialize (%s)"),
	   get_status_string (status));

  if (amd_dbgapi_async_event_handler != nullptr)
    delete_async_event_handler (&amd_dbgapi_async_event_handler);
}

/* Implementation of `_wave_id' variable.  */

static struct value *
amd_dbgapi_wave_id_make_value (struct gdbarch *gdbarch, struct internalvar *var,
			 void *ignore)
{
  if (ptid_is_gpu (inferior_ptid))
    {
      amd_dbgapi_wave_id_t wave_id = get_amd_dbgapi_wave_id (inferior_ptid);
      uint32_t group_ids[3], wave_in_group;

      if (amd_dbgapi_wave_get_info (wave_id,
				    AMD_DBGAPI_WAVE_INFO_WORKGROUP_COORD,
				    sizeof (group_ids), &group_ids)
	    == AMD_DBGAPI_STATUS_SUCCESS
	  && amd_dbgapi_wave_get_info (
	       wave_id, AMD_DBGAPI_WAVE_INFO_WAVE_NUMBER_IN_WORKGROUP,
	       sizeof (wave_in_group), &wave_in_group)
	       == AMD_DBGAPI_STATUS_SUCCESS)
	{
	  std::string wave_id_str
	    = string_printf ("(%d,%d,%d)/%d", group_ids[0], group_ids[1],
			     group_ids[2], wave_in_group);

	  return value_cstring (wave_id_str.data (), wave_id_str.length () + 1,
				builtin_type (gdbarch)->builtin_char);
	}
    }

  return allocate_value (builtin_type (gdbarch)->builtin_void);
}

static const struct internalvar_funcs amd_dbgapi_wave_id_funcs
  = { amd_dbgapi_wave_id_make_value, NULL };

static void
show_precise_memory_mode (struct ui_file *file, int from_tty,
			  struct cmd_list_element *c, const char *value)
{
  struct amd_dbgapi_inferior_info *info = get_amd_dbgapi_inferior_info ();

  fprintf_filtered (file,
		    _ ("AMDGPU precise memory access reporting is %s "
		       "(currently %s).\n"),
		    info->precise_memory.requested ? "on" : "off",
		    info->precise_memory.enabled ? "enabled" : "disabled");
}

static void
set_precise_memory_mode (bool value)
{
  struct amd_dbgapi_inferior_info *info = get_amd_dbgapi_inferior_info ();

  info->precise_memory.requested = value;

  if (info->process_id != AMD_DBGAPI_PROCESS_NONE)
    {
      amd_dbgapi_memory_precision_t memory_precision
	= info->precise_memory.requested ? AMD_DBGAPI_MEMORY_PRECISION_PRECISE
					 : AMD_DBGAPI_MEMORY_PRECISION_NONE;

      if (set_process_memory_precision (info->process_id, memory_precision))
	info->precise_memory.enabled = info->precise_memory.requested;
      else
	warning (
	  _ ("AMDGPU precise memory access reporting could not be enabled."));
    }
}

static bool
get_precise_memory_mode ()
{
  struct amd_dbgapi_inferior_info *info = get_amd_dbgapi_inferior_info ();
  return info->precise_memory.requested;
}

static bool
get_effective_precise_memory_mode ()
{
  amd_dbgapi_inferior_info *info = get_amd_dbgapi_inferior_info ();
  return info->precise_memory.enabled;
};

static const char *
get_dbgapi_library_file_path ()
{
  Dl_info dl_info{};
  if (!dladdr ((void*) amd_dbgapi_get_version, &dl_info))
    return "";
  return dl_info.dli_fname;
}

static void
show_dbgapi_version (const char *args, int from_tty)
{
  uint32_t major, minor, patch;
  amd_dbgapi_get_version (&major, &minor, &patch);

  printf_filtered ("%p[ROCdbgapi %d.%d.%d (%s)%p]\nLoaded from `%ps'\n",
                   version_style.style ().ptr (), major, minor, patch,
		   amd_dbgapi_get_build_name (), nullptr,
		   styled_string (file_name_style.style (),
				  get_dbgapi_library_file_path ()));
}

/* List of set/show amdgpu commands.  */
struct cmd_list_element *set_amdgpu_list;
struct cmd_list_element *show_amdgpu_list;

/* List of set/show debug amdgpu commands.  */
struct cmd_list_element *set_debug_amdgpu_list;
struct cmd_list_element *show_debug_amdgpu_list;

constexpr const char *debug_amdgpu_log_level_enums[]
  = { /* [AMD_DBGAPI_LOG_LEVEL_NONE] = */ "off",
      /* [AMD_DBGAPI_LOG_LEVEL_FATAL_ERROR] = */ "error",
      /* [AMD_DBGAPI_LOG_LEVEL_WARNING] = */ "warning",
      /* [AMD_DBGAPI_LOG_LEVEL_INFO] = */ "info",
      /* [AMD_DBGAPI_LOG_LEVEL_TRACE] = */ "trace",
      /* [AMD_DBGAPI_LOG_LEVEL_VERBOSE] = */ "verbose",
      nullptr };

static const char *debug_amdgpu_log_level
  = debug_amdgpu_log_level_enums[AMD_DBGAPI_LOG_LEVEL_WARNING];

static amd_dbgapi_log_level_t
get_debug_amdgpu_log_level ()
{
  for (size_t pos = 0; debug_amdgpu_log_level_enums[pos]; ++pos)
    if (debug_amdgpu_log_level == debug_amdgpu_log_level_enums[pos])
      return static_cast<amd_dbgapi_log_level_t> (pos);

  error (_ ("Invalid log level: %s"), debug_amdgpu_log_level);
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
  struct ui_out *uiout = current_uiout;
  amd_dbgapi_status_t status;

  amd_dbgapi_agent_id_t current_agent_id;
  if ((uiout->is_mi_like_p () && args != nullptr && *args != '\0')
      || !ptid_is_gpu (inferior_ptid)
      || amd_dbgapi_wave_get_info (get_amd_dbgapi_wave_id (inferior_ptid),
				   AMD_DBGAPI_WAVE_INFO_AGENT,
				   sizeof (current_agent_id),
				   &current_agent_id)
	   != AMD_DBGAPI_STATUS_SUCCESS)
    current_agent_id = AMD_DBGAPI_AGENT_NONE;

  {
    gdb::optional<ui_out_emit_list> list_emitter;
    gdb::optional<ui_out_emit_table> table_emitter;

    std::vector<std::pair<inferior *, std::vector<amd_dbgapi_agent_id_t>>>
      all_filtered_agents;

    for (inferior *inf : all_inferiors ())
      {
	amd_dbgapi_process_id_t process_id = get_amd_dbgapi_process_id (inf);
	amd_dbgapi_agent_id_t *agent_list;
	size_t agent_count;

	if (process_id == AMD_DBGAPI_PROCESS_NONE)
	  continue;

	if (amd_dbgapi_process_agent_list (process_id, &agent_count,
					   &agent_list, nullptr)
	    != AMD_DBGAPI_STATUS_SUCCESS)
	  continue;

	std::vector<amd_dbgapi_agent_id_t> filtered_agents;
	std::copy_if (&agent_list[0], &agent_list[agent_count],
		      std::back_inserter (filtered_agents),
		      [=] (auto agent_id)
		      {
			return tid_is_in_list (args, current_inferior ()->num,
					       inf->num, agent_id.handle);
		      });

	all_filtered_agents.emplace_back (inf, std::move (filtered_agents));
	xfree (agent_list);
      }

    if (uiout->is_mi_like_p ())
      list_emitter.emplace (uiout, "agents");
    else
      {
	size_t n_agents{ 0 }, max_id_width{ 0 }, max_target_id_width{ 0 },
	  max_name_width{ 0 }, max_architecture_width{ 0 };

	for (auto &&value : all_filtered_agents)
	  {
	    inferior *inf = value.first;
	    auto &agents = value.second;

	    for (auto &&agent_id : agents)
	      {
		/* id  */
		max_id_width
		  = std::max (max_id_width,
			      (show_inferior_qualified_tids ()
				   || uiout->is_mi_like_p ()
				 ? string_printf ("%d.%ld", inf->num,
						  agent_id.handle)
				 : string_printf ("%ld", agent_id.handle))
				.size ());

		/* target id  */
		max_target_id_width
		  = std::max (max_target_id_width,
			      agent_target_id_string (agent_id).size ());

		/* architecture  */
		amd_dbgapi_architecture_id_t architecture_id;

		status = amd_dbgapi_agent_get_info (
		  agent_id, AMD_DBGAPI_AGENT_INFO_ARCHITECTURE,
		  sizeof (architecture_id), &architecture_id);

		if (status == AMD_DBGAPI_STATUS_SUCCESS)
		  {
		    char *architecture_name;
		    if ((status = amd_dbgapi_architecture_get_info (
			   architecture_id, AMD_DBGAPI_ARCHITECTURE_INFO_NAME,
			   sizeof (architecture_name), &architecture_name))
			!= AMD_DBGAPI_STATUS_SUCCESS)
		      error (_("amd_dbgapi_architecture_get_info failed (%s)"),
			     get_status_string (status));

		    max_architecture_width
		      = std::max (max_architecture_width,
				  strlen (architecture_name));
		    xfree (architecture_name);
		  }
		else if (status != AMD_DBGAPI_STATUS_ERROR_NOT_AVAILABLE)
		  error (_("amd_dbgapi_agent_get_info failed (%s)"),
			 get_status_string (status));

		/* name  */
		char *agent_name;
		if ((status
		     = amd_dbgapi_agent_get_info (agent_id,
						  AMD_DBGAPI_AGENT_INFO_NAME,
						  sizeof (agent_name),
						  &agent_name))
		    != AMD_DBGAPI_STATUS_SUCCESS)
		  error (_("amd_dbgapi_agent_get_info failed (%s)"),
			 get_status_string (status));

		max_name_width
		  = std::max (max_name_width, strlen (agent_name));
		xfree (agent_name);

		++n_agents;
	      }
	  }

	if (!n_agents)
	  {
	    if (args == nullptr || *args == '\0')
	      uiout->message (_ ("No agents are currently active.\n"));
	    else
	      uiout->message (_ ("No active agents match '%s'.\n"), args);
	    return;
	  }

	/* Header:  */
	table_emitter.emplace (uiout, 9, n_agents, "InfoRocmAgentsTable");

	uiout->table_header (1, ui_left, "current", "");
	uiout->table_header (std::max (2ul, max_id_width), ui_left, "id",
			     "Id");
	uiout->table_header (5, ui_left, "state", "State");
	uiout->table_header (std::max (9ul, max_target_id_width), ui_left,
			     "target-id", "Target Id");
	uiout->table_header (std::max (12ul, max_architecture_width), ui_left,
			     "architecture", "Architecture");
	uiout->table_header (std::max (11ul, max_name_width), ui_left, "name",
			     "Device Name");
	uiout->table_header (5, ui_left, "cores", "Cores");
	uiout->table_header (7, ui_left, "threads", "Threads");
	uiout->table_header (8, ui_left, "location", "Location");
	uiout->table_body ();
      }

    /* Rows:  */
    for (auto &&value : all_filtered_agents)
      {
	inferior *inf = value.first;
	auto &agents = value.second;

	std::sort (agents.begin (), agents.end (),
		   [] (amd_dbgapi_agent_id_t lhs, amd_dbgapi_agent_id_t rhs)
		   { return lhs.handle < rhs.handle; });

	for (auto &&agent_id : agents)
	  {
	    ui_out_emit_tuple tuple_emitter (uiout, nullptr);

	    /* current  */
	    if (!uiout->is_mi_like_p ())
	      {
		if (agent_id == current_agent_id)
		  uiout->field_string ("current", "*");
		else
		  uiout->field_skip ("current");
	      }

	    /* id-in-th  */
	    uiout->field_string ("id",
				 (show_inferior_qualified_tids ()
				      || uiout->is_mi_like_p ()
				    ? string_printf ("%d.%ld", inf->num,
						     agent_id.handle)
				    : string_printf ("%ld", agent_id.handle))
				   .c_str ());

	    /* supported  */
	    amd_dbgapi_agent_state_t agent_state;
	    if ((status
		 = amd_dbgapi_agent_get_info (agent_id,
					      AMD_DBGAPI_AGENT_INFO_STATE,
					      sizeof (agent_state),
					      &agent_state))
		!= AMD_DBGAPI_STATUS_SUCCESS)
	      error (_("amd_dbgapi_agent_get_info failed (%s)"),
		     get_status_string (status));
	    switch (agent_state)
	      {
	      case AMD_DBGAPI_AGENT_STATE_SUPPORTED:
		uiout->field_string ("state", "A");
		break;
	      case AMD_DBGAPI_AGENT_STATE_NOT_SUPPORTED:
		uiout->field_string ("state", "U");
		break;
	      }

	    /* target_id  */
	    uiout->field_string ("target-id",
				 agent_target_id_string (agent_id));

	    /* architecture  */
	    amd_dbgapi_architecture_id_t architecture_id;

	    status
	      = amd_dbgapi_agent_get_info (agent_id,
					   AMD_DBGAPI_AGENT_INFO_ARCHITECTURE,
					   sizeof (architecture_id),
					   &architecture_id);

	    if (status == AMD_DBGAPI_STATUS_SUCCESS)
	      {
		char *architecture_name;
		if ((status = amd_dbgapi_architecture_get_info (
		       architecture_id, AMD_DBGAPI_ARCHITECTURE_INFO_NAME,
		       sizeof (architecture_name), &architecture_name))
		    != AMD_DBGAPI_STATUS_SUCCESS)
		  error (_("amd_dbgapi_architecture_get_info failed (%s)"),
			 get_status_string (status));

		uiout->field_string ("architecture", architecture_name);
		xfree (architecture_name);
	      }
	    else if (status == AMD_DBGAPI_STATUS_ERROR_NOT_AVAILABLE)
	      uiout->field_string ("architecture", "unknown");
	    else
	      error (_("amd_dbgapi_agent_get_info failed (%s)"),
		     get_status_string (status));

	    /* name  */
	    char *agent_name;
	    if ((status
		 = amd_dbgapi_agent_get_info (agent_id,
					      AMD_DBGAPI_AGENT_INFO_NAME,
					      sizeof (agent_name),
					      &agent_name))
		!= AMD_DBGAPI_STATUS_SUCCESS)
	      error (_("amd_dbgapi_agent_get_info failed (%s)"),
		     get_status_string (status));

	    uiout->field_string ("name", agent_name);
	    xfree (agent_name);

	    /* cores  */
	    size_t cores;
	    if ((status = amd_dbgapi_agent_get_info (
		   agent_id, AMD_DBGAPI_AGENT_INFO_EXECUTION_UNIT_COUNT,
		   sizeof (cores), &cores))
		!= AMD_DBGAPI_STATUS_SUCCESS)
	      error (_("amd_dbgapi_agent_get_info failed (%s)"),
		     get_status_string (status));

	    uiout->field_signed ("cores", cores);

	    /* threads  */
	    size_t threads;
	    if ((status = amd_dbgapi_agent_get_info (
		   agent_id,
		   AMD_DBGAPI_AGENT_INFO_MAX_WAVES_PER_EXECUTION_UNIT,
		   sizeof (threads), &threads))
		!= AMD_DBGAPI_STATUS_SUCCESS)
	      error (_("amd_dbgapi_agent_get_info failed (%s)"),
		     get_status_string (status));

	    uiout->field_signed ("threads", cores * threads);

	    /* location  */
	    uint16_t location;
	    if ((status
		 = amd_dbgapi_agent_get_info (agent_id,
					      AMD_DBGAPI_AGENT_INFO_PCI_SLOT,
					      sizeof (location), &location))
		!= AMD_DBGAPI_STATUS_SUCCESS)
	      error (_("amd_dbgapi_agent_get_info failed (%s)"),
		     get_status_string (status));

	    uiout->field_string ("location",
				 string_printf ("%02x:%02x.%d",
						(location >> 8) & 0xFF,
						(location >> 3) & 0x1F,
						location & 0x7));

	    uiout->text ("\n");
	  }
      }
  }

  if (uiout->is_mi_like_p () && current_agent_id != AMD_DBGAPI_AGENT_NONE)
    uiout->field_signed ("current-agent-id", current_agent_id.handle);

  gdb_flush (gdb_stdout);
}

static struct cmd_list_element *queue_list;

static void
info_queues_command (const char *args, int from_tty)
{
  struct gdbarch *gdbarch = target_gdbarch ();
  struct ui_out *uiout = current_uiout;
  amd_dbgapi_status_t status;

  amd_dbgapi_queue_id_t current_queue_id;
  if ((uiout->is_mi_like_p () && args != nullptr && *args != '\0')
      || !ptid_is_gpu (inferior_ptid)
      || amd_dbgapi_wave_get_info (get_amd_dbgapi_wave_id (inferior_ptid),
				   AMD_DBGAPI_WAVE_INFO_QUEUE,
				   sizeof (current_queue_id),
				   &current_queue_id)
	   != AMD_DBGAPI_STATUS_SUCCESS)
    current_queue_id = AMD_DBGAPI_QUEUE_NONE;

  {
    gdb::optional<ui_out_emit_list> list_emitter;
    gdb::optional<ui_out_emit_table> table_emitter;

    std::vector<std::pair<inferior *, std::vector<amd_dbgapi_queue_id_t>>>
      all_filtered_queues;

    for (inferior *inf : all_inferiors ())
      {
	amd_dbgapi_process_id_t process_id = get_amd_dbgapi_process_id (inf);
	amd_dbgapi_queue_id_t *queue_list;
	size_t queue_count;

	if (process_id == AMD_DBGAPI_PROCESS_NONE)
	  continue;

	if (amd_dbgapi_process_queue_list (process_id, &queue_count,
					   &queue_list, nullptr)
	    != AMD_DBGAPI_STATUS_SUCCESS)
	  continue;

	std::vector<amd_dbgapi_queue_id_t> filtered_queues;
	std::copy_if (&queue_list[0], &queue_list[queue_count],
		      std::back_inserter (filtered_queues),
		      [=] (auto queue_id)
		      {
			return tid_is_in_list (args, current_inferior ()->num,
					       inf->num, queue_id.handle);
		      });

	all_filtered_queues.emplace_back (inf, std::move (filtered_queues));
	xfree (queue_list);
      }

    if (uiout->is_mi_like_p ())
      list_emitter.emplace (uiout, "queues");
    else
      {
	size_t n_queues{ 0 }, max_target_id_width{ 0 };

	for (auto &&value : all_filtered_queues)
	  {
	    auto &queues = value.second;

	    for (auto &&queue_id : queues)
	      {
		/* target id  */
		max_target_id_width
		  = std::max (max_target_id_width,
			      queue_target_id_string (queue_id).size ());

		++n_queues;
	      }
	  }

	if (!n_queues)
	  {
	    if (args == nullptr || *args == '\0')
	      uiout->message (_ ("No queues are currently active.\n"));
	    else
	      uiout->message (_ ("No active queues match '%s'.\n"), args);
	    return;
	  }

	/* Header:  */
	table_emitter.emplace (uiout, 8, n_queues, "InfoRocmQueuesTable");

	uiout->table_header (1, ui_left, "current", "");
	uiout->table_header (show_inferior_qualified_tids () ? 6 : 4, ui_left,
			     "id", "Id");
	uiout->table_header (std::max (9ul, max_target_id_width), ui_left,
			     "target-id", "Target Id");
	uiout->table_header (12, ui_left, "type", "Type");
	uiout->table_header (6, ui_left, "read", "Read");
	uiout->table_header (6, ui_left, "write", "Write");
	uiout->table_header (8, ui_left, "size", "Size");
	uiout->table_header (2 + (gdbarch_ptr_bit (gdbarch) / 4), ui_left,
			     "addr", "Address");
	uiout->table_body ();
      }

    /* Rows:  */
    for (auto &&value : all_filtered_queues)
      {
	inferior *inf = value.first;
	auto &queues = value.second;

	std::sort (queues.begin (), queues.end (),
		   [] (amd_dbgapi_queue_id_t lhs, amd_dbgapi_queue_id_t rhs)
		   { return lhs.handle < rhs.handle; });

	for (auto &&queue_id : queues)
	  {
	    ui_out_emit_tuple tuple_emitter (uiout, nullptr);

	    if (!uiout->is_mi_like_p ())
	      {
		/* current  */
		if (queue_id == current_queue_id)
		  uiout->field_string ("current", "*");
		else
		  uiout->field_skip ("current");
	      }

	    /* id  */
	    uiout->field_string ("id",
				 (show_inferior_qualified_tids ()
				      || uiout->is_mi_like_p ()
				    ? string_printf ("%d.%ld", inf->num,
						     queue_id.handle)
				    : string_printf ("%ld", queue_id.handle))
				   .c_str ());

	    /* target-id  */
	    uiout->field_string ("target-id",
				 queue_target_id_string (queue_id));

	    /* type  */
	    amd_dbgapi_os_queue_type_t type;
	    if ((status
		 = amd_dbgapi_queue_get_info (queue_id,
					      AMD_DBGAPI_QUEUE_INFO_TYPE,
					      sizeof (type), &type))
		!= AMD_DBGAPI_STATUS_SUCCESS)
	      error (_("amd_dbgapi_queue_get_info failed (%s)"),
		     get_status_string (status));

	    uiout->field_string (
	      "type",
	      [type] ()
	      {
		switch (type)
		  {
		  case AMD_DBGAPI_OS_QUEUE_TYPE_HSA_AQL:
		    return "HSA";
		  case AMD_DBGAPI_OS_QUEUE_TYPE_AMD_PM4:
		    return "PM4";
		  case AMD_DBGAPI_OS_QUEUE_TYPE_AMD_SDMA:
		    return "DMA";
		  case AMD_DBGAPI_OS_QUEUE_TYPE_AMD_SDMA_XGMI:
		    return "XGMI";
		  case AMD_DBGAPI_OS_QUEUE_TYPE_UNKNOWN:
		  default:
		    return "Unknown";
		  }
	      }());

	    /* read, write  */
	    amd_dbgapi_os_queue_packet_id_t read, write;
	    size_t unused;
	    if ((status
		 = amd_dbgapi_queue_packet_list (queue_id, &read, &write,
						 &unused, nullptr))
		== AMD_DBGAPI_STATUS_SUCCESS)
	      {
		uiout->field_signed ("read", read);
		uiout->field_signed ("write", write);
	      }
	    else if (status == AMD_DBGAPI_STATUS_ERROR_NOT_SUPPORTED)
	      {
		uiout->field_skip ("read");
		uiout->field_skip ("write");
	      }
	    else
	      error (_("amd_dbgapi_queue_get_info failed (%s)"),
		     get_status_string (status));

	    /* size  */
	    amd_dbgapi_size_t size;
	    if ((status
		 = amd_dbgapi_queue_get_info (queue_id,
					      AMD_DBGAPI_QUEUE_INFO_SIZE,
					      sizeof (size), &size))
		!= AMD_DBGAPI_STATUS_SUCCESS)
	      error (_("amd_dbgapi_queue_get_info failed (%s)"),
		     get_status_string (status));

	    uiout->field_signed ("size", size);

	    /* addr */
	    amd_dbgapi_global_address_t addr;
	    if ((status
		 = amd_dbgapi_queue_get_info (queue_id,
					      AMD_DBGAPI_QUEUE_INFO_ADDRESS,
					      sizeof (addr), &addr))
		!= AMD_DBGAPI_STATUS_SUCCESS)
	      error (_("amd_dbgapi_queue_get_info failed (%s)"),
		     get_status_string (status));

	    uiout->field_core_addr ("addr", gdbarch, addr);

	    uiout->text ("\n");
	  }
      }
  }

  if (uiout->is_mi_like_p () && current_queue_id != AMD_DBGAPI_QUEUE_NONE)
    uiout->field_signed ("current-queue-id", current_queue_id.handle);

  gdb_flush (gdb_stdout);
}

static void
queue_find_command (const char *arg, int from_tty)
{
  if (!arg || !*arg)
    error (_ ("Command requires an argument."));

  const char *tmp = re_comp (arg);
  if (tmp)
    error (_ ("Invalid regexp (%s): %s"), tmp, arg);

  size_t matches = 0;
  for (inferior *inf : all_inferiors ())
    {
      amd_dbgapi_process_id_t process_id = get_amd_dbgapi_process_id (inf);
      amd_dbgapi_queue_id_t *queue_list;
      size_t queue_count;

      if (process_id == AMD_DBGAPI_PROCESS_NONE)
	continue;

      if (amd_dbgapi_process_queue_list (process_id, &queue_count, &queue_list,
					 nullptr)
	  != AMD_DBGAPI_STATUS_SUCCESS)
	continue;

      std::vector<amd_dbgapi_queue_id_t> queues (&queue_list[0],
						 &queue_list[queue_count]);

      xfree (queue_list);

      for (auto &&queue_id : queues)
	{
	  std::string target_id = queue_target_id_string (queue_id);
	  if (re_exec (target_id.c_str ()))
	    {
	      printf_filtered (_ ("Queue %ld has Target Id '%s'\n"),
			       queue_id.handle, target_id.c_str ());
	      ++matches;
	    }
	}
    }

  if (!matches)
    printf_filtered (_ ("No queues match '%s'\n"), arg);
}

template <typename T>
static std::string
ndim_string (uint32_t dims, T *sizes)
{
  std::stringstream ss;

  ss << "[";
  for (uint32_t i = 0; i < dims; ++i)
    {
      if (i)
	ss << ",";
      ss << sizes[i];
    }
  ss << "]";

  return ss.str ();
}

template <typename T>
static size_t
num_digits (T value)
{
  size_t digits{ value < 0 ? 1 : 0 };

  if (!value)
    return 1;

  while (value)
    {
      value /= 10;
      ++digits;
    }
  return digits;
}

static struct cmd_list_element *dispatch_list;

struct info_dispatches_opts
{
  bool full = false;
};

static const gdb::option::option_def info_dispatches_option_defs[] = {
  gdb::option::flag_option_def<info_dispatches_opts>{
    "full",
    [] (info_dispatches_opts *opts) { return &opts->full; },
    N_ ("Display all fields."),
  },
};

static inline gdb::option::option_def_group
make_info_dispatches_options_def_group (info_dispatches_opts *opts)
{
  return { { info_dispatches_option_defs }, opts };
}

static void
info_dispatches_command_completer (struct cmd_list_element *ignore,
				   completion_tracker &tracker,
				   const char *text, const char *word_ignored)
{
  const auto grp = make_info_dispatches_options_def_group (nullptr);

  if (gdb::option::
	complete_options (tracker, &text,
			  gdb::option::PROCESS_OPTIONS_UNKNOWN_IS_ERROR, grp))
    return;

  /* Convenience to let the user know what the option can accept.  */
  if (*text == '\0')
    {
      gdb::option::complete_on_all_options (tracker, grp);
      /* Keep this "ID" in sync with what "help info threads"
	 says.  */
      tracker.add_completion (make_unique_xstrdup ("ID"));
    }
}

static void
info_dispatches_command (const char *args, int from_tty)
{
  struct gdbarch *gdbarch = target_gdbarch ();
  struct ui_out *uiout = current_uiout;
  amd_dbgapi_status_t status;

  info_dispatches_opts opts;
  auto grp = make_info_dispatches_options_def_group (&opts);
  gdb::option::process_options (&args,
				gdb::option::PROCESS_OPTIONS_UNKNOWN_IS_ERROR,
				grp);

  amd_dbgapi_dispatch_id_t current_dispatch_id;
  if ((uiout->is_mi_like_p () && args != nullptr && *args != '\0')
      || !ptid_is_gpu (inferior_ptid)
      || amd_dbgapi_wave_get_info (get_amd_dbgapi_wave_id (inferior_ptid),
				   AMD_DBGAPI_WAVE_INFO_DISPATCH,
				   sizeof (current_dispatch_id),
				   &current_dispatch_id)
	   != AMD_DBGAPI_STATUS_SUCCESS)
    current_dispatch_id = AMD_DBGAPI_DISPATCH_NONE;

  {
    gdb::optional<ui_out_emit_list> list_emitter;
    gdb::optional<ui_out_emit_table> table_emitter;

    std::vector<std::pair<inferior *, std::vector<amd_dbgapi_dispatch_id_t>>>
      all_filtered_dispatches;

    /* We'll be switching inferiors temporarily below.  */
    inferior *curr_inferior = current_inferior ();
    scoped_restore_current_thread restore_thread;

    for (inferior *inf : all_inferiors ())
      {
	amd_dbgapi_process_id_t process_id = get_amd_dbgapi_process_id (inf);
	amd_dbgapi_dispatch_id_t *dispatch_list;
	size_t dispatch_count;

	if (process_id == AMD_DBGAPI_PROCESS_NONE)
	  continue;

	if (amd_dbgapi_process_dispatch_list (process_id, &dispatch_count,
					      &dispatch_list, nullptr)
	    != AMD_DBGAPI_STATUS_SUCCESS)
	  continue;

	std::vector<amd_dbgapi_dispatch_id_t> filtered_dispatches;
	std::copy_if (&dispatch_list[0], &dispatch_list[dispatch_count],
		      std::back_inserter (filtered_dispatches),
		      [=] (auto dispatch_id)
		      {
			return tid_is_in_list (args, curr_inferior->num,
					       inf->num, dispatch_id.handle);
		      });

	all_filtered_dispatches.emplace_back (inf,
					      std::move (filtered_dispatches));
	xfree (dispatch_list);
      }

    if (uiout->is_mi_like_p ())
      list_emitter.emplace (uiout, "dispatches");
    else
      {
	size_t n_dispatches{ 0 }, max_target_id_width{ 0 },
	  max_grid_width{ 0 }, max_workgroup_width{ 0 },
	  max_address_spaces_width{ 0 };

	for (auto &&value : all_filtered_dispatches)
	  {
	    auto &dispatches = value.second;

	    for (auto &&dispatch_id : dispatches)
	      {
		/* target id  */
		max_target_id_width
		  = std::max (max_target_id_width,
			      dispatch_target_id_string (dispatch_id).size ());

		/* grid  */
		uint32_t dims;
		if ((status = amd_dbgapi_dispatch_get_info (
		       dispatch_id, AMD_DBGAPI_DISPATCH_INFO_GRID_DIMENSIONS,
		       sizeof (dims), &dims))
		    != AMD_DBGAPI_STATUS_SUCCESS)
		  error (_("amd_dbgapi_dispatch_get_info failed (%s)"),
			 get_status_string (status));

		uint32_t grid_sizes[3];
		if ((status = amd_dbgapi_dispatch_get_info (
		       dispatch_id, AMD_DBGAPI_DISPATCH_INFO_GRID_SIZES,
		       sizeof (grid_sizes), &grid_sizes[0]))
		    != AMD_DBGAPI_STATUS_SUCCESS)
		  error (_("amd_dbgapi_dispatch_get_info failed (%s)"),
			 get_status_string (status));

		max_grid_width
		  = std::max (max_grid_width,
			      ndim_string (dims, grid_sizes).size ());

		/* workgroup  */
		uint16_t work_group_sizes[3];
		if ((status = amd_dbgapi_dispatch_get_info (
		       dispatch_id, AMD_DBGAPI_DISPATCH_INFO_WORKGROUP_SIZES,
		       sizeof (work_group_sizes), &work_group_sizes[0]))
		    != AMD_DBGAPI_STATUS_SUCCESS)
		  error (_("amd_dbgapi_dispatch_get_info failed (%s)"),
			 get_status_string (status));

		max_workgroup_width
		  = std::max (max_workgroup_width,
			      ndim_string (dims, work_group_sizes).size ());

		/* address-spaces  */
		amd_dbgapi_size_t shared_size, private_size;
		if ((status = amd_dbgapi_dispatch_get_info (
		       dispatch_id,
		       AMD_DBGAPI_DISPATCH_INFO_GROUP_SEGMENT_SIZE,
		       sizeof (shared_size), &shared_size))
		    != AMD_DBGAPI_STATUS_SUCCESS)
		  error (_("amd_dbgapi_dispatch_get_info failed (%s)"),
			 get_status_string (status));
		if ((status = amd_dbgapi_dispatch_get_info (
		       dispatch_id,
		       AMD_DBGAPI_DISPATCH_INFO_PRIVATE_SEGMENT_SIZE,
		       sizeof (private_size), &private_size))
		    != AMD_DBGAPI_STATUS_SUCCESS)
		  error (_("amd_dbgapi_dispatch_get_info failed (%s)"),
			 get_status_string (status));

		max_address_spaces_width
		  = std::max (max_address_spaces_width,
			      string_printf ("Shared(%ld), Private(%ld)",
					     shared_size, private_size)
				.size ());

		++n_dispatches;
	      }
	  }

	if (!n_dispatches)
	  {
	    if (args == nullptr || *args == '\0')
	      uiout->message (_ ("No dispatches are currently active.\n"));
	    else
	      uiout->message (_ ("No active dispatches match '%s'.\n"), args);
	    return;
	  }

	/* Header:  */
	table_emitter.emplace (uiout, opts.full ? 11 : 7, n_dispatches,
			       "InfoRocmDispatchesTable");
	size_t addr_width = 2 + (gdbarch_ptr_bit (gdbarch) / 4);

	uiout->table_header (1, ui_left, "current", "");
	uiout->table_header (show_inferior_qualified_tids () ? 6 : 4, ui_left,
			     "id", "Id");
	uiout->table_header (std::max (9ul, max_target_id_width), ui_left,
			     "target-id", "Target Id");
	uiout->table_header (std::max (4ul, max_grid_width), ui_left, "grid",
			     "Grid");
	uiout->table_header (std::max (9ul, max_workgroup_width), ui_left,
			     "workgroup", "Workgroup");
	uiout->table_header (7, ui_left, "fence", "Fence");
	if (opts.full)
	  {
	    uiout->table_header (std::max (14ul, max_address_spaces_width),
				 ui_left, "address-spaces", "Address Spaces");
	    uiout->table_header (std::max (17ul, addr_width), ui_left,
				 "kernel-desc", "Kernel Descriptor");
	    uiout->table_header (std::max (11ul, addr_width), ui_left,
				 "kernel-args", "Kernel Args");
	    uiout->table_header (std::max (17ul, addr_width), ui_left,
				 "completion", "Completion Signal");
	  }
	uiout->table_header (1, ui_left, "kernel-function", "Kernel Function");
	uiout->table_body ();
      }

    /* Rows:  */
    for (auto &&value : all_filtered_dispatches)
      {
	inferior *inf = value.first;
	auto &dispatches = value.second;

	std::sort (dispatches.begin (), dispatches.end (),
		   [] (amd_dbgapi_dispatch_id_t lhs,
		       amd_dbgapi_dispatch_id_t rhs)
		   { return lhs.handle < rhs.handle; });

	/* Switch the inferior since we are doing a symbol lookup in this
	   inferior's program space.  */
	switch_to_inferior_no_thread (inf);

	for (auto &&dispatch_id : dispatches)
	  {
	    ui_out_emit_tuple tuple_emitter (uiout, nullptr);

	    if (!uiout->is_mi_like_p ())
	      {
		/* current  */
		if (dispatch_id == current_dispatch_id)
		  uiout->field_string ("current", "*");
		else
		  uiout->field_skip ("current");
	      }

	    /* id  */
	    uiout->field_string ("id", (show_inferior_qualified_tids ()
					    || uiout->is_mi_like_p ()
					  ? string_printf ("%d.%ld", inf->num,
							   dispatch_id.handle)
					  : string_printf ("%ld",
							   dispatch_id.handle))
					 .c_str ());

	    /* target-id  */
	    uiout->field_string ("target-id",
				 dispatch_target_id_string (dispatch_id));

	    /* grid  */
	    uint32_t dims;
	    if ((status = amd_dbgapi_dispatch_get_info (
		   dispatch_id, AMD_DBGAPI_DISPATCH_INFO_GRID_DIMENSIONS,
		   sizeof (dims), &dims))
		!= AMD_DBGAPI_STATUS_SUCCESS)
	      error (_("amd_dbgapi_dispatch_get_info failed (%s)"),
		     get_status_string (status));

	    uint32_t grid_sizes[3];
	    if ((status = amd_dbgapi_dispatch_get_info (
		   dispatch_id, AMD_DBGAPI_DISPATCH_INFO_GRID_SIZES,
		   sizeof (grid_sizes), &grid_sizes[0]))
		!= AMD_DBGAPI_STATUS_SUCCESS)
	      error (_("amd_dbgapi_dispatch_get_info failed (%s)"),
		     get_status_string (status));

	    uiout->field_string ("grid", ndim_string (dims, grid_sizes));

	    /* workgroup  */
	    uint16_t work_group_sizes[3];
	    if ((status = amd_dbgapi_dispatch_get_info (
		   dispatch_id, AMD_DBGAPI_DISPATCH_INFO_WORKGROUP_SIZES,
		   sizeof (work_group_sizes), &work_group_sizes[0]))
		!= AMD_DBGAPI_STATUS_SUCCESS)
	      error (_("amd_dbgapi_dispatch_get_info failed (%s)"),
		     get_status_string (status));

	    uiout->field_string ("workgroup",
				 ndim_string (dims, work_group_sizes));

	    /* fence  */
	    amd_dbgapi_dispatch_barrier_t barrier;
	    if ((status = amd_dbgapi_dispatch_get_info (
		   dispatch_id, AMD_DBGAPI_DISPATCH_INFO_BARRIER,
		   sizeof (barrier), &barrier))
		!= AMD_DBGAPI_STATUS_SUCCESS)
	      error (_("amd_dbgapi_dispatch_get_info failed (%s)"),
		     get_status_string (status));

	    amd_dbgapi_dispatch_fence_scope_t acquire, release;
	    if ((status = amd_dbgapi_dispatch_get_info (
		   dispatch_id, AMD_DBGAPI_DISPATCH_INFO_ACQUIRE_FENCE,
		   sizeof (acquire), &acquire))
		!= AMD_DBGAPI_STATUS_SUCCESS)
	      error (_("amd_dbgapi_dispatch_get_info failed (%s)"),
		     get_status_string (status));
	    if ((status = amd_dbgapi_dispatch_get_info (
		   dispatch_id, AMD_DBGAPI_DISPATCH_INFO_RELEASE_FENCE,
		   sizeof (release), &release))
		!= AMD_DBGAPI_STATUS_SUCCESS)
	      error (_("amd_dbgapi_dispatch_get_info failed (%s)"),
		     get_status_string (status));

	    std::stringstream ss;
	    if (barrier == AMD_DBGAPI_DISPATCH_BARRIER_PRESENT)
	      ss << "B";

	    if (barrier && acquire)
	      ss << "|";

	    if (acquire == AMD_DBGAPI_DISPATCH_FENCE_SCOPE_AGENT)
	      ss << "Aa";
	    else if (acquire == AMD_DBGAPI_DISPATCH_FENCE_SCOPE_SYSTEM)
	      ss << "As";

	    if ((barrier | acquire) && release)
	      ss << "|";

	    if (release == AMD_DBGAPI_DISPATCH_FENCE_SCOPE_AGENT)
	      ss << "Ra";
	    else if (release == AMD_DBGAPI_DISPATCH_FENCE_SCOPE_SYSTEM)
	      ss << "Rs";

	    uiout->field_string ("fence", ss.str ());

	    if (opts.full || uiout->is_mi_like_p ())
	      {
		/* address-spaces  */
		amd_dbgapi_size_t shared_size, private_size;
		if ((status = amd_dbgapi_dispatch_get_info (
		       dispatch_id,
		       AMD_DBGAPI_DISPATCH_INFO_GROUP_SEGMENT_SIZE,
		       sizeof (shared_size), &shared_size))
		    != AMD_DBGAPI_STATUS_SUCCESS)
		  error (_("amd_dbgapi_dispatch_get_info failed (%s)"),
			 get_status_string (status));
		if ((status = amd_dbgapi_dispatch_get_info (
		       dispatch_id,
		       AMD_DBGAPI_DISPATCH_INFO_PRIVATE_SEGMENT_SIZE,
		       sizeof (private_size), &private_size))
		    != AMD_DBGAPI_STATUS_SUCCESS)
		  error (_("amd_dbgapi_dispatch_get_info failed (%s)"),
			 get_status_string (status));

		uiout
		  ->field_string ("address-spaces",
				  string_printf ("Shared(%ld), Private(%ld)",
						 shared_size, private_size));

		/* kernel-desc  */
		amd_dbgapi_global_address_t kernel_desc;
		if ((status = amd_dbgapi_dispatch_get_info (
		       dispatch_id,
		       AMD_DBGAPI_DISPATCH_INFO_KERNEL_DESCRIPTOR_ADDRESS,
		       sizeof (kernel_desc), &kernel_desc))
		    != AMD_DBGAPI_STATUS_SUCCESS)
		  error (_("amd_dbgapi_dispatch_get_info failed (%s)"),
			 get_status_string (status));

		uiout->field_core_addr ("kernel-desc", gdbarch, kernel_desc);

		/* kernel-args  */
		amd_dbgapi_global_address_t kernel_args;
		if (
		  (status = amd_dbgapi_dispatch_get_info (
		     dispatch_id,
		     AMD_DBGAPI_DISPATCH_INFO_KERNEL_ARGUMENT_SEGMENT_ADDRESS,
		     sizeof (kernel_args), &kernel_args))
		  != AMD_DBGAPI_STATUS_SUCCESS)
		  error (_("amd_dbgapi_dispatch_get_info failed (%s)"),
			 get_status_string (status));

		uiout->field_core_addr ("kernel-args", gdbarch, kernel_args);

		/* completion  */
		amd_dbgapi_global_address_t completion;
		if ((status = amd_dbgapi_dispatch_get_info (
		       dispatch_id,
		       AMD_DBGAPI_DISPATCH_INFO_KERNEL_COMPLETION_ADDRESS,
		       sizeof (completion), &completion))
		    != AMD_DBGAPI_STATUS_SUCCESS)
		  error (_("amd_dbgapi_dispatch_get_info failed (%s)"),
			 get_status_string (status));

		if (completion || uiout->is_mi_like_p ())
		  uiout->field_core_addr ("completion", gdbarch, completion);
		else
		  uiout->field_string ("completion", "(nil)");
	      }

	    /* kernel-function  */
	    amd_dbgapi_global_address_t kernel_code;
	    if ((status = amd_dbgapi_dispatch_get_info (
		   dispatch_id,
		   AMD_DBGAPI_DISPATCH_INFO_KERNEL_CODE_ENTRY_ADDRESS,
		   sizeof (kernel_code), &kernel_code))
		!= AMD_DBGAPI_STATUS_SUCCESS)
	      error (_("amd_dbgapi_dispatch_get_info failed (%s)"),
		     get_status_string (status));

	    auto msymbol
	      = lookup_minimal_symbol_by_pc_section (kernel_code, nullptr);
	    if (msymbol.minsym && !uiout->is_mi_like_p ())
	      uiout->field_string ("kernel-function",
				   msymbol.minsym->print_name (),
				   function_name_style.style ());
	    else
	      uiout->field_core_addr ("kernel-function", gdbarch, kernel_code);

	    uiout->text ("\n");
	  }
      }
  }

  if (uiout->is_mi_like_p ()
      && current_dispatch_id != AMD_DBGAPI_DISPATCH_NONE)
    uiout->field_signed ("current-dispatch-id", current_dispatch_id.handle);

  gdb_flush (gdb_stdout);
}

/* Dump out a table of address spaces for the current architecture.  */

static void
address_spaces_dump (struct gdbarch *gdbarch, struct ui_file *file)
{
  if (!gdbarch_address_spaces_p (gdbarch))
    return;

  auto address_spaces = gdbarch_address_spaces (gdbarch);

  fprintf_unfiltered (file, " Name\n");

  for (const auto &address_space : address_spaces)
    {
      fprintf_unfiltered (file, " %-10s\n", address_space.name.get ());
    }
}

/* Dump out a table of register groups for the current architecture.  */

static void
maintenance_print_address_spaces (const char *args, int from_tty)
{
  address_spaces_dump (get_current_arch (), gdb_stdout);
}

static void
dispatch_find_command (const char *arg, int from_tty)
{
  amd_dbgapi_status_t status;

  if (!arg || !*arg)
    error (_ ("Command requires an argument."));

  const char *tmp = re_comp (arg);
  if (tmp)
    error (_ ("Invalid regexp (%s): %s"), tmp, arg);

  /* We'll be switching inferiors temporarily below.  */
  scoped_restore_current_thread restore_thread;

  size_t matches = 0;
  for (inferior *inf : all_inferiors ())
    {
      amd_dbgapi_process_id_t process_id = get_amd_dbgapi_process_id (inf);
      amd_dbgapi_dispatch_id_t *dispatch_list;
      size_t dispatch_count;

      if (process_id == AMD_DBGAPI_PROCESS_NONE)
	continue;

      if (amd_dbgapi_process_dispatch_list (process_id, &dispatch_count,
					    &dispatch_list, nullptr)
	  != AMD_DBGAPI_STATUS_SUCCESS)
	continue;

      std::vector<amd_dbgapi_dispatch_id_t>
	dispatches (&dispatch_list[0], &dispatch_list[dispatch_count]);

      xfree (dispatch_list);

      /* Switch the inferior since we are doing a symbol lookup in this
	 inferior's program space.  */
      switch_to_inferior_no_thread (inf);

      for (auto &&dispatch_id : dispatches)
	{
	  std::string target_id = dispatch_target_id_string (dispatch_id);
	  if (re_exec (target_id.c_str ()))
	    {
	      printf_filtered (_ ("Dispatch %ld has Target Id '%s'\n"),
			       dispatch_id.handle, target_id.c_str ());
	      ++matches;
	    }

	  amd_dbgapi_global_address_t kernel_code;
	  if ((status = amd_dbgapi_dispatch_get_info (
		 dispatch_id,
		 AMD_DBGAPI_DISPATCH_INFO_KERNEL_CODE_ENTRY_ADDRESS,
		 sizeof (kernel_code), &kernel_code))
	      != AMD_DBGAPI_STATUS_SUCCESS)
	    error (_("amd_dbgapi_dispatch_get_info failed (%s)"),
		   get_status_string (status));

	  auto msymbol
	    = lookup_minimal_symbol_by_pc_section (kernel_code, nullptr);
	  if (msymbol.minsym && re_exec (msymbol.minsym->print_name ()))
	    {
	      printf_filtered (_ ("Dispatch %ld has Kernel Function '%s'\n"),
			       dispatch_id.handle,
			       msymbol.minsym->print_name ());
	      ++matches;
	    }
	}
    }

  if (!matches)
    printf_filtered (_ ("No dispatches match '%s'\n"), arg);
}

/* -Wmissing-prototypes */
extern initialize_file_ftype _initialize_amd_dbgapi_target;

void
_initialize_amd_dbgapi_target ()
{
  /* Make sure the loaded debugger library version is greater than or equal to
     the one used to build GDB.  */
  uint32_t major, minor, patch;
  amd_dbgapi_get_version (&major, &minor, &patch);
  if (major != AMD_DBGAPI_VERSION_MAJOR || minor < AMD_DBGAPI_VERSION_MINOR)
    error (_("amd-dbgapi library version mismatch, got %d.%d.%d, need %d.%d+"),
	   major, minor, patch, AMD_DBGAPI_VERSION_MAJOR,
	   AMD_DBGAPI_VERSION_MINOR);

  /* Initialize the AMD Debugger API.  */
  amd_dbgapi_status_t status = amd_dbgapi_initialize (&dbgapi_callbacks);
  if (status != AMD_DBGAPI_STATUS_SUCCESS)
    error (_("amd-dbgapi failed to initialize (%s)"),
	   get_status_string (status));

  /* Set the initial log level.  */
  amd_dbgapi_set_log_level (get_debug_amdgpu_log_level ());

  /* Install observers.  */
  gdb::observers::breakpoint_created.attach (amd_dbgapi_target_breakpoint_fixup,
					     "amd-dbgapi");
  gdb::observers::inferior_created.attach (amd_dbgapi_target_inferior_created,
					   "amd-dbgapi");
  gdb::observers::inferior_cloned.attach (amd_dbgapi_target_inferior_cloned,
					  "amd-dbgapi");
  gdb::observers::signal_received.attach (amd_dbgapi_target_signal_received,
					  "amd-dbgapi");
  gdb::observers::normal_stop.attach (amd_dbgapi_target_normal_stop, "amd-dbgapi");
  gdb::observers::inferior_execd.attach (amd_dbgapi_inferior_execd, "amd-dbgapi");

  create_internalvar_type_lazy ("_wave_id", &amd_dbgapi_wave_id_funcs, NULL);

  add_basic_prefix_cmd ("amdgpu", no_class,
			_ ("Generic command for setting amdgpu flags."),
			&set_amdgpu_list, 0, &setlist);

  add_show_prefix_cmd ("amdgpu", no_class,
		       _ ("Generic command for showing amdgpu flags."),
		       &show_amdgpu_list, 0, &showlist);

  set_show_commands cmds
    = add_setshow_boolean_cmd ("precise-memory", no_class,
			       _ ("Set precise-memory mode."),
			       _ ("Show precise-memory mode."), _ ("\
If on, precise memory reporting is enabled if/when the inferior is running.\n\
If off (default), precise memory reporting is disabled."),
			       set_precise_memory_mode, get_precise_memory_mode,
			       show_precise_memory_mode,
			       &set_amdgpu_list, &show_amdgpu_list);

  cmds.show->var->set_effective_value_getter<bool>
    (get_effective_precise_memory_mode);

  add_cmd ("version", no_set_class, show_dbgapi_version,
	   _("Show the ROCdbgapi library version and build information."),
	   &show_amdgpu_list);

  add_basic_prefix_cmd ("amdgpu", no_class,
			_ ("Generic command for setting amdgpu debugging "
			   "flags."),
			&set_debug_amdgpu_list, 0, &setdebuglist);

  add_show_prefix_cmd ("amdgpu", no_class,
		       _ ("Generic command for showing amdgpu debugging "
			  "flags."),
		       &show_debug_amdgpu_list, 0, &showdebuglist);

  add_setshow_enum_cmd ("log-level", class_maintenance,
			debug_amdgpu_log_level_enums, &debug_amdgpu_log_level,
			_ ("Set the amdgpu log level."),
			_ ("Show the amdgpu log level."),
			_ (
			  "off     == no logging is enabled\n"
			  "error   == fatal errors are reported\n"
			  "warning == fatal errors and warnings are reported\n"
			  "info    == fatal errors, warnings, and info "
			  "messages are reported\n"
			  "trace   == fatal errors, warnings, info, and "
			  "API tracing messages are reported\n"
			  "verbose == all messages are reported"),
			set_debug_amdgpu_log_level,
			show_debug_amdgpu_log_level, &set_debug_amdgpu_list,
			&show_debug_amdgpu_list);

  add_cmd ("agents", class_info, info_agents_command,
	   _ ("(Display currently active heterogeneous agents.\n\
Usage: info agents [ID]...\n\
\n\
If ID is given, it is a space-separated list of IDs of agents to display.\n\
Otherwise, all agents are displayed."),
	   &infolist);

  add_basic_prefix_cmd ("queue", class_run,
			_ ("Commands that operate on heterogeneous queues."),
			&queue_list, 0, &cmdlist);

  add_cmd ("find", class_run, queue_find_command, _ ("\
Find heterogeneous queues that match a regular expression.\n\
Usage: queue find REGEXP\n\
Will display queue IDs whose Target ID matches REGEXP."),
	   &queue_list);

  add_cmd ("queues", class_info, info_queues_command,
	   _ ("Display currently active heterogeneous queues.\n\
Usage: info queues [ID]...\n\
\n\
If ID is given, it is a space-separated list of IDs of queues to display.\n\
Otherwise, all queues are displayed."),
	   &infolist);

  add_basic_prefix_cmd ("dispatch", class_run,
			_ ("Commands that operate on heterogeneous "
			   "dispatches."),
			&dispatch_list, 0, &cmdlist);

  add_cmd ("find", class_run, dispatch_find_command, _ ("\
Find heterogeneous dispatches that match a regular expression.\n\
Usage: dispatch find REGEXP\n\
Will display dispatch IDs whose Target ID or Kernel Function matches REGEXP."),
	   &dispatch_list);

  const auto info_dispatches_opts
    = make_info_dispatches_options_def_group (nullptr);

  static std::string info_dispatches_help
    = gdb::option::build_help (_ ("\
Display currently active heterogeneous dispatches.\n\
Usage: info dispatches [ID]...\n\
\n\
Options:\n\
%OPTIONS%\n\n\
If ID is given, it is a space-separated list of IDs of dispatches to display.\n\
Otherwise, all dispatches are displayed."),
			       info_dispatches_opts);

  auto *c = add_info ("dispatches", info_dispatches_command,
		      info_dispatches_help.c_str ());
  set_cmd_completer_handle_brkchars (c, info_dispatches_command_completer);

  add_cmd ("address-spaces", class_maintenance,
	   maintenance_print_address_spaces, _("\
Displays the address space names supported by \
each target architecture.\n\
Takes an optional file parameter."),
	   &maintenanceprintlist);
}
