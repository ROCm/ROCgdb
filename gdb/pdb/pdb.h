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

#if defined (PDB_FORMAT_AVAILABLE)

/* Try PDB initialization for OBJFILE from the COFF symbol reader.
   Return true when the PDB that the executable's RSDS record names was
   loaded.  Return false when there is no such record, or when the PDB
   cannot be found, read or matched, so the caller can look for debug
   information elsewhere.  A mismatched or malformed PDB is reported (a
   warning for the main objfile, a complaint otherwise) and nothing is
   read from it.  */

extern bool pdb_initialize_objfile (struct objfile *objfile);

#else /* PDB_FORMAT_AVAILABLE */

/* This configuration has no PDB reader: no PDB is ever loaded.  */

static inline bool
pdb_initialize_objfile (struct objfile *)
{
  return false;
}

#endif /* PDB_FORMAT_AVAILABLE */

#endif /* GDB_PDB_PDB_H */
