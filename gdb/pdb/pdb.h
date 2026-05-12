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

/* Try PDB initialization for OBJFILE from the COFF symbol reader.
   Return false when no supported RSDS record with a usable PDB filename
   is found, allowing the caller to try another debug format.

   Return true once such an RSDS record is found, even if loading is skipped
   or the PDB cannot be loaded or has a mismatched GUID/age.  True therefore
   selects the PDB path; it does not guarantee that symbols were loaded.
   Malformed required PDB data can raise an error instead of returning.
   A successful load installs lazy lookup, or expands modules for readnow.  */

extern bool pdb_initialize_objfile (struct objfile *objfile);

/* namespace pdb */
}

/* GDB_PDB_PDB_H */
#endif
