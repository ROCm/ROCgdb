/* GDB HIP testing mode.

   Copyright (C) 2021 Free Software Foundation, Inc.
   Copyright (C) 2021 Advanced Micro Devices, Inc. All rights reserved.

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

/* If requested by setting GDB_HIP_TEST_MODE=1 in the environment,
   enable GDB HIP testing mode.  This mode renames "gdb_hip_test_main"
   to "main", and "main" to "__gdb_host_Main", to better hide the host
   code and provide the illusion that the kernel/device's main is
   really called main.  */

#include "defs.h"
#include "gdb-hip-test-mode.h"
#include "objfiles.h"

/* 0=disabled.  1=enabled.  2=debug.  */
static int gdb_hip_test_mode = 0;

/* What to rename the driver/host's main to.  Note the uppercase "M"
   so that "rbreak main" doesn't catch it.  */
static const char host_name[] = "__gdb_host_Main";

/* What to rename the kernel/device's gdb_test_hip_main to.  */
static const char device_name[] = "main";

/* These are global so that we're sure to compute strlen only
   once.  */
static const gdb::string_view host_name_sv = host_name;
static const gdb::string_view device_name_sv = device_name;

gdb::string_view
gdb_hip_test_mode_frob_names (struct objfile *objfile, gdb::string_view name)
{
  if (!gdb_hip_test_mode)
    return name;

  if (objfile->obfd->arch_info->arch != bfd_arch_amdgcn
      && name == "main")
    {
      if (gdb_hip_test_mode > 1)
	fprintf_unfiltered (gdb_stdlog, "frob: got host main\n");
      return host_name_sv;
    }
  else if (objfile->obfd->arch_info->arch != bfd_arch_amdgcn
	   && name == "_ZN3lld3elf12LinkerDriver4mainEN4llvm8ArrayRefIPKcEE")
    {
      /* /opt/rocm/lib/libamd_comgr.so.1 calls
	   lld::elf::LinkerDriver::main(llvm::ArrayRef<char const*>)
	 before we reach the device's main.  */
      if (gdb_hip_test_mode > 1)
	fprintf_unfiltered (gdb_stdlog, "frob: got host %.*s\n",
			    (int) name.size (), name.data ());
      /* Same, with uppercase "Main".  */
      return "_ZN3lld3elf12LinkerDriver4MainEN4llvm8ArrayRefIPKcEE";
    }
  else if (name.find ("gdb_hip_test_main") != gdb::string_view::npos)
    {
      if (gdb_hip_test_mode > 1)
	fprintf_unfiltered (gdb_stdlog, "frob: got %.*s\n",
			    (int) name.size (), name.data ());
      return device_name_sv;
    }

  return name;
}

const char *
gdb_hip_test_mode_frob_names (struct objfile *objfile, const char *name)
{
  if (!gdb_hip_test_mode)
    return name;
  return gdb_hip_test_mode_frob_names (objfile, gdb::string_view (name)).data ();
}

void _initialize_gdb_hip_test_mode ();

void
_initialize_gdb_hip_test_mode ()
{
  const char *p = getenv ("GDB_HIP_TEST_MODE");
  if (p != nullptr)
    gdb_hip_test_mode = atoi (p);
  if (gdb_hip_test_mode > 1)
    fprintf_unfiltered (gdb_stdlog, "gdb hip test mode enabled, mode=%d\n",
			gdb_hip_test_mode);
}
