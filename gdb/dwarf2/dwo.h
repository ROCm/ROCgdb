/* DWARF 2 debugging format support for GDB.

   Copyright (C) 1994-2026 Free Software Foundation, Inc.

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

#ifndef GDB_DWARF2_DWO_H
#define GDB_DWARF2_DWO_H

#include "dwarf2/section.h"
#include "gdb_bfd.h"
#include "gdbsupport/unordered_set.h"
#include "hashtab.h"

/* These sections are what may appear in a (real or virtual) DWO file.  */

struct dwo_sections
{
  struct dwarf2_section_info abbrev;
  struct dwarf2_section_info line;
  struct dwarf2_section_info loc;
  struct dwarf2_section_info loclists;
  struct dwarf2_section_info macinfo;
  struct dwarf2_section_info macro;
  struct dwarf2_section_info rnglists;
  struct dwarf2_section_info str;
  struct dwarf2_section_info str_offsets;
  /* In the case of a virtual DWO file, these two are unused.  */
  std::vector<dwarf2_section_info> infos;
  std::vector<dwarf2_section_info> types;
};

/* CUs/TUs in DWP/DWO files.  */

struct dwo_unit
{
  /* Backlink to the containing struct dwo_file.  */
  struct dwo_file *dwo_file = nullptr;

  /* The "id" that distinguishes this CU/TU.
     .debug_info calls this "dwo_id", .debug_types calls this "signature".
     Since signatures came first, we stick with it for consistency.  */
  ULONGEST signature = 0;

  /* The section this CU/TU lives in, in the DWO file.  */
  dwarf2_section_info *section = nullptr;

  /* This is set if SECTION is owned by this dwo_unit.  */
  dwarf2_section_info_up section_holder;

  /* Same as dwarf2_per_cu::{sect_off,length} but in the DWO section.  */
  sect_offset sect_off {};
  unsigned int length = 0;

  /* For types, offset in the type's DIE of the type defined by this TU.  */
  cu_offset type_offset_in_tu;
};

using dwo_unit_up = std::unique_ptr<dwo_unit>;

/* Hash function for dwo_unit objects, based on the signature.  */

struct dwo_unit_hash
{
  using is_transparent = void;

  std::size_t operator() (ULONGEST signature) const noexcept
  { return signature; }

  std::size_t operator() (const dwo_unit_up &unit) const noexcept
  { return (*this) (unit->signature); }
};

/* Equal function for dwo_unit objects, based on the signature.

   The signature is assumed to be unique within the DWO file.  So while object
   file CU dwo_id's always have the value zero, that's OK, assuming each object
   file DWO file has only one CU, and that's the rule for now.  */

struct dwo_unit_eq
{
  using is_transparent = void;

  bool operator() (ULONGEST sig, const dwo_unit_up  &unit) const noexcept
  { return sig == unit->signature; }

  bool operator() (const dwo_unit_up &a, const dwo_unit_up &b) const noexcept
  { return (*this) (a->signature, b); }
};

/* Set of dwo_unit object, using their signature as identity.  */

using dwo_unit_set = gdb::unordered_set<dwo_unit_up, dwo_unit_hash, dwo_unit_eq>;

/* Data for one DWO file.

   This includes virtual DWO files (a virtual DWO file is a DWO file as it
   appears in a DWP file).  DWP files don't really have DWO files per se -
   comdat folding of types "loses" the DWO file they came from, and from
   a high level view DWP files appear to contain a mass of random types.
   However, to maintain consistency with the non-DWP case we pretend DWP
   files contain virtual DWO files, and we assign each TU with one virtual
   DWO file (generally based on the line and abbrev section offsets -
   a heuristic that seems to work in practice).  */

struct dwo_file
{
  dwo_file () = default;
  DISABLE_COPY_AND_ASSIGN (dwo_file);

  /* Look for a type unit with signature SIGNATURE in this dwo_file.

     Return nullptr if not found.  */
  dwo_unit *find_tu (ULONGEST signature) const
  {
    if (auto it = this->tus.find (signature); it != this->tus.end ())
      return it->get ();

    return nullptr;
  }

  /* The DW_AT_GNU_dwo_name or DW_AT_dwo_name attribute.
     For virtual DWO files the name is constructed from the section offsets
     of abbrev,line,loc,str_offsets so that we combine virtual DWO files
     from related CU+TUs.  */
  std::string dwo_name;

  /* The DW_AT_comp_dir attribute.  */
  const char *comp_dir = nullptr;

  /* The bfd, when the file is open.  Otherwise this is NULL.
     This is unused(NULL) for virtual DWO files where we use dwp_file.dbfd.  */
  gdb_bfd_ref_ptr dbfd;

  /* The sections that make up this DWO file.
     Remember that for virtual DWO files in DWP V2 or DWP V5, these are virtual
     sections (for lack of a better name).  */
  struct dwo_sections sections {};

  /* The CUs in the file.

     Multiple CUs per DWO are supported as an extension to handle LLVM's Link
     Time Optimization output (where multiple source files may be compiled into
     a single object/dwo pair).  */
  dwo_unit_set cus;

  /* Table of TUs in the file.  */
  dwo_unit_set tus;
};

using dwo_file_up = std::unique_ptr<dwo_file>;

/* This is used when looking up entries in a dwo_file_set.  */

struct dwo_file_search
{
  /* Name of the DWO to look for.  */
  const char *dwo_name;

  /* Compilation directory to look for.  */
  const char *comp_dir;
};

/* Hash function for dwo_file objects, using their dwo_name and comp_dir as
   identity.  */

struct dwo_file_hash
{
  using is_transparent = void;

  std::size_t operator() (const dwo_file_search &search) const noexcept
  {
    hashval_t hash = htab_hash_string (search.dwo_name);

    if (search.comp_dir != nullptr)
      hash += htab_hash_string (search.comp_dir);

    return hash;
  }

  std::size_t operator() (const dwo_file_up &file) const noexcept
  {
    return (*this) ({ file->dwo_name.c_str (), file->comp_dir });
  }
};

/* Equal function for dwo_file objects, using their dwo_name and comp_dir as
   identity.  */

struct dwo_file_eq
{
  using is_transparent = void;

  bool operator() (const dwo_file_search &search,
		   const dwo_file_up &dwo_file) const noexcept
  {
    if (search.dwo_name != dwo_file->dwo_name)
      return false;

    if (search.comp_dir == nullptr || dwo_file->comp_dir == nullptr)
      return search.comp_dir == dwo_file->comp_dir;

    return streq (search.comp_dir, dwo_file->comp_dir);
  }

  bool operator() (const dwo_file_up &a, const dwo_file_up &b) const noexcept
  {
    return (*this) ({ a->dwo_name.c_str (), a->comp_dir }, b);
  }
};

/* Set of dwo_file objects, using their dwo_name and comp_dir as identity.  */

using dwo_file_up_set
  = gdb::unordered_set<dwo_file_up, dwo_file_hash, dwo_file_eq>;

#endif /* GDB_DWARF2_DWO_H */
