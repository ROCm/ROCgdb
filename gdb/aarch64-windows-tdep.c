/* Target-dependent code for Windows AArch64.

   Copyright (C) 2025-2026 Free Software Foundation, Inc.

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

#include "gdbarch.h"
#include "aarch64-tdep.h"
#include "arch-utils.h"
#include "regset.h"
#include "windows-tdep.h"

/* Windows uses the various BRK instruction variants for special operations,
   and BRK #0xf000 triggers a breakpoint exception in the debugger.  */
constexpr gdb_byte aarch64_windows_breakpoint[] = {0x00, 0x00, 0x3e, 0xd4};

typedef BP_MANIPULATION (aarch64_windows_breakpoint) aarch64_w_breakpoint;

/* gdbarch initialization for Windows on AArch64.  */

static void
aarch64_windows_init_abi (struct gdbarch_info info, struct gdbarch *gdbarch)
{
  set_gdbarch_ps_regnum (gdbarch, AARCH64_CPSR_REGNUM);

  set_gdbarch_breakpoint_kind_from_pc (gdbarch,
				       aarch64_w_breakpoint::kind_from_pc);
  set_gdbarch_sw_breakpoint_from_kind (gdbarch,
				       aarch64_w_breakpoint::bp_from_kind);

  /* Usually the arm BRK instruction triggers with the PC still at the
     instruction, but Windows increments the PC before notifying the
     debugger.  */
  set_gdbarch_decr_pc_after_break (gdbarch, 4);

  windows_init_abi (info, gdbarch);
}

static gdb_osabi
aarch64_windows_osabi_sniffer (bfd *abfd)
{
  const char *target_name = bfd_get_target (abfd);

  if (!streq (target_name, "pei-aarch64-little"))
    return GDB_OSABI_UNKNOWN;

  return GDB_OSABI_WINDOWS;
}

INIT_GDB_FILE (aarch64_windows_tdep)
{
  gdbarch_register_osabi (bfd_arch_aarch64, 0, GDB_OSABI_WINDOWS,
			  aarch64_windows_init_abi);

  gdbarch_register_osabi_sniffer (bfd_arch_aarch64, bfd_target_coff_flavour,
				  aarch64_windows_osabi_sniffer);
}
