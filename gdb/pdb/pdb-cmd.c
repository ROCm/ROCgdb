/* PDB debugging format support for GDB.

   Copyright (C) 2026 Free Software Foundation, Inc.
   Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

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
   along with this program.  If not, see <http://www.gnu.org/licenses/>.

   This file implements various PDB related GDB info commands.  */
#include "cli/cli-cmds.h"
#include "objfiles.h"
#include "pdb/pdb-internal.h"
#include "progspace.h"

namespace pdb
{

/* 'info pdb-loaded-files' command.  */

static void
pdb_info_loaded_files_command (const char *args, int /*from_tty*/)
{
  if (args != nullptr && *args != '\0')
    {
      gdb_printf ("Usage: info pdb-loaded-files\n");
      return;
    }

  bool found = false;
  for (objfile &objf : current_program_space->objfiles ())
    {
      const pdb_per_objfile *pdb = get_pdb_per_objfile (&objf);
      if (pdb == nullptr || pdb->pdb_file_path.empty ())
	continue;
      gdb_printf ("%s\n", pdb->pdb_file_path.c_str ());
      found = true;
    }

  if (!found)
    gdb_printf ("No PDB loaded.\n");
}

} /* namespace pdb */

/* Register PDB maintenance info commands.  */
INIT_GDB_FILE (pdb_info)
{
  using namespace pdb;
  add_cmd ("pdb-loaded-files", class_maintenance,
	   pdb_info_loaded_files_command,
	   _("List currently loaded PDB file paths."), &maintenanceinfolist);
}
