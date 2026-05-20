/* Target-dependent code for GNU/Linux, architecture independent.

   Copyright (C) 2009-2026 Free Software Foundation, Inc.

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

#ifndef GDB_LINUX_TDEP_H
#define GDB_LINUX_TDEP_H

#include "bfd.h"
#include "displaced-stepping.h"
#include "solib.h"

struct inferior;
struct regcache;

/* Return true if ADDRESS is within the boundaries of a page mapped with
   memory tagging protection.  */
bool linux_address_in_memtag_page (CORE_ADDR address);

extern enum gdb_signal linux_gdb_signal_from_target (struct gdbarch *gdbarch,
						     int signal);

extern int linux_gdb_signal_to_target (struct gdbarch *gdbarch,
				       enum gdb_signal signal);

/* Default GNU/Linux implementation of `displaced_step_location', as
   defined in gdbarch.h.  Determines the entry point from AT_ENTRY in
   the target auxiliary vector.  */
extern CORE_ADDR linux_displaced_step_location (struct gdbarch *gdbarch);


/* Implementation of gdbarch_displaced_step_prepare.  */

extern displaced_step_prepare_status linux_displaced_step_prepare
  (gdbarch *arch, thread_info *thread, CORE_ADDR &displaced_pc);

/* Implementation of gdbarch_displaced_step_finish.  */

extern displaced_step_finish_status linux_displaced_step_finish
  (gdbarch *arch, thread_info *thread, const target_waitstatus &status);

/* Implementation of gdbarch_displaced_step_copy_insn_closure_by_addr.  */

extern const displaced_step_copy_insn_closure *
  linux_displaced_step_copy_insn_closure_by_addr
    (inferior *inf, CORE_ADDR addr);

/* Implementation of gdbarch_displaced_step_restore_all_in_ptid.  */

extern void linux_displaced_step_restore_all_in_ptid (inferior *parent_inf,
						      ptid_t ptid);

extern void linux_init_abi (struct gdbarch_info info, struct gdbarch *gdbarch,
			    int num_disp_step_buffers);

extern bool linux_is_uclinux ();

/* Fetch the AT_HWCAP entry from auxv data AUXV.  Use TARGET and GDBARCH to
   parse auxv entries.

   On error, 0 is returned.  */
extern CORE_ADDR linux_get_hwcap (const std::optional<gdb::byte_vector> &auxv,
				  struct target_ops *target, gdbarch *gdbarch);

/* Same as the above, but obtain all the inputs from the current inferior.  */

extern CORE_ADDR linux_get_hwcap ();

/* Fetch the AT_HWCAP2 entry from auxv data AUXV.  Use TARGET and GDBARCH to
   parse auxv entries.

   On error, 0 is returned.  */
extern CORE_ADDR linux_get_hwcap2 (const std::optional<gdb::byte_vector> &auxv,
				   struct target_ops *target, gdbarch *gdbarch);

/* Same as the above, but obtain all the inputs from the current inferior.  */

extern CORE_ADDR linux_get_hwcap2 ();

/* Fetch the AT_HWCAP3 entry from auxv data AUXV.  Use TARGET and GDBARCH to
   parse auxv entries.

   On error, 0 is returned.  */
extern CORE_ADDR linux_get_hwcap3 (const std::optional<gdb::byte_vector> &auxv,
				   struct target_ops *target, gdbarch *gdbarch);

/* Same as the above, but obtain all the inputs from the current inferior.  */

extern CORE_ADDR linux_get_hwcap3 ();

/* Fetch the AT_HWCAP4 entry from auxv data AUXV.  Use TARGET and GDBARCH to
   parse auxv entries.

   On error, 0 is returned.  */
extern CORE_ADDR linux_get_hwcap4 (const std::optional<gdb::byte_vector> &auxv,
				   struct target_ops *target, gdbarch *gdbarch);

/* Same as the above, but obtain all the inputs from the current inferior.  */

extern CORE_ADDR linux_get_hwcap4 ();

/* Returns true if ADDR belongs to a shadow stack memory range.  If this
   is the case, assign the shadow stack memory range to RANGE
   [start_address, end_address).  */

extern bool linux_address_in_shadow_stack_mem_range
  (CORE_ADDR addr, std::pair<CORE_ADDR, CORE_ADDR> *range);

namespace gdb {

/* Maps each siginfo_type::key to the corresponding field-access expression
   in $_siginfo.

   Keep the order of key values synchronized with the entries in get()'s
   paths array.  Each key is used directly as an array index.  */

struct siginfo_type
{
  /* Identifies a field within siginfo_t that may be referenced by name.  */
  enum class key
  {
    siginfo_signo = 0,
    siginfo_errno,
    siginfo_code,

    /* SIGILL, SIGFPE, SIGSEGV, SIGBUS, SIGTRAP, SIGEMT */
    siginfo_addr,
    siginfo_trapno,
    siginfo_addr_lsb,
    siginfo_lower,
    siginfo_upper,
    siginfo_pkey,
    siginfo_perf_data,
    siginfo_perf_type,
    siginfo_perf_flags,

    /* Sentinel used to determine the number of mapped fields.  */
    SIGINFO_ATTR_END
  };

  /* Return the $_siginfo access expression associated with ATTR_.

     ATTR_ must be a valid si_* key other than SIGINFO_ATTR_END.  The array
     order must exactly match the declaration order of the keys above.  */
  static constexpr const char *get (key attr_)
  {
    const char *paths[static_cast<size_t> (key::SIGINFO_ATTR_END)] = {
      "$_siginfo.si_signo",
      "$_siginfo.si_errno",
      "$_siginfo.si_code",

      /* SIGILL, SIGFPE, SIGSEGV, SIGBUS, SIGTRAP, SIGEMT */
      "$_siginfo._sifields._sigfault.si_addr",
      "$_siginfo._sifields._sigfault.si_trapno",
      "$_siginfo._sifields._sigfault.si_addr_lsb",
      "$_siginfo._sifields._sigfault._addr_bnd.si_lower",
      "$_siginfo._sifields._sigfault._addr_bnd.si_upper",
      "$_siginfo._sifields._sigfault._addr_pkey.si_pkey",
      "$_siginfo._sifields._sigfault._perf.si_perf_data",
      "$_siginfo._sifields._sigfault._perf.si_perf_type",
      "$_siginfo._sifields._sigfault._perf.si_perf_flags",
    };
    return paths[static_cast<size_t> (attr_)];
  }
};

} /* namespace gdb */

#endif /* GDB_LINUX_TDEP_H */
