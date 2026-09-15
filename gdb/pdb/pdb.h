/* PDB debugging format support for GDB - Public header.

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
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

/* Public PDB API consumed by code outside the PDB reader.  */

#ifndef GDB_PDB_PDB_H
#define GDB_PDB_PDB_H

struct objfile;

namespace pdb
{

/* Main entry point for reading PDB debug info for the objfile.
   Reads the PDB file and processes its debug information and associates
   it with the objfile. Returns true if PDB data was found and processed.
   Called from coff_symfile_read, analogous to dwarf2_initialize_objfile.  */
extern bool pdb_initialize_objfile (struct objfile *objfile);

} // namespace pdb

#endif /* GDB_PDB_PDB_H */
