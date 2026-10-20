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

   This file implements PDB file path discovery for PE/COFF executables
   and the conversion of MSYS2 source paths recorded in a PDB.  */

#include "objfiles.h"
#include "pdb/pdb-internal.h"
#include "gdb_bfd.h"

#include "coff/internal.h"
#include "libcoff.h"

#include <string.h>
#include <algorithm>
#include <array>

/* "RSDS" magic — PDB 7.0 CodeView signature.  Defined in
   <coff/pe.h>, but including it here conflicts with the COFF
   headers already included above.  */
inline constexpr auto CVINFO_PDB70_CVSIGNATURE = 0x53445352;

namespace pdb
{

/* Path conversion debug statements, at "set debug pdb-read 2".  */
#define pdb_cvt_printf(fmt, ...)                                      \
  debug_prefixed_printf_cond (pdb_read_debug >= 2, "pdb cvt", fmt, \
			      ##__VA_ARGS__)

/* Convert Unix-style paths to Windows paths as GDB has problems with MSVC
   compiled PDBs that may contain paths like /home/rocm/... when built under
   MSYS2.  This conversion is used before storing the file paths to lineinfo.
   When path is converted we check for the presence of the converted path and if
   it doesn't exist we return the original path.  Path issues to solve:
   1. Under MSYS2, the linker prepends the work dir to the file path.  E.g.
      /home/user/test/C:\Program Files\...\include/__msvc_ostream.hpp*
      We strip the work dir: E.g. /<WHATEVER>/<DRIVE>:\ -> <DRIVE>:\
   2. The pure unix style path - e.g. /home/user/test/file.cpp
      Here we need to find the MSYS2 Windows root and prepend it to the path.
      E.g. C:/msys2/home/user/test/file.cpp.
      The MSYS2 root is not derivable: neither the configure-time gdbdir nor
      the gdb executable dir gives it for every layout.
      For example:
	C:/msys2/mingw64/WHATEVER/gdb -> Root is C:/msys2/mingw64
	C:/msys2/WHATEVER/gdb -> Root is C:/msys2
      but C:/mypath/share/gdb yields nothing.  MSYS2_ROOT must therefore be
      set when a pure unix path is found.
      If path under MSYS2_ROOT or MSYS2_ROOT is not set we will try some well
      known paths:
       a) E.g. <DRIVE>/WHATEVER - search for <DRIVE>:/WHATEVER
       b) Assume MSYS2_ROOT is <DRIVE>:/msys64/mingw64 or <DRIVE>:/msys2  */

static std::string
pdb_resolve_path (const char *path)
{
  if (path == nullptr || path[0] == '\0')
    return std::string ();

  /* Non-Unix paths are returned as-is.  */
  if (path[0] != '/')
    return std::string (path);

  pdb_cvt_printf ("converting path: '%s'", path);

  /* Case 1: Embedded Windows path in a Unix prefix.
     E.g. /home/user/test/C:\Program Files\...\include/file.c
     Search for ':' to find a drive letter pattern /X:\ or /X:/ and strip
     everything before the drive letter.  */
  std::string spath (path);
  if (size_t pos = spath.find (':'); pos != std::string::npos && pos >= 2)
    {
      if (spath[pos - 2] == '/' && isalpha ((unsigned char) spath[pos - 1])
	  && pos + 1 < spath.size ()
	  && (spath[pos + 1] == '\\' || spath[pos + 1] == '/'))
	{
	  std::string result = spath.substr (pos - 1);
	  if (access (result.c_str (), F_OK) == 0)
	    {
	      pdb_cvt_printf ("Converted: '%s' -> '%s'", path,
			      result.c_str ());
	      return result;
	    }
	}

      /* If file is not found return the original string.
	 TODO: decide whether the later schemes should still be tried
	 once a drive letter was found.  */
      if (pos != std::string::npos)
	return spath;
    }

  /* Case 2: Unix-style path.  */

  /* First, try the MSYS2_ROOT environment variable.  */
  const char *env_root = getenv ("MSYS2_ROOT");
  if (env_root != nullptr && env_root[0] != '\0')
    {
      std::string result = std::string (env_root) + path;
      /* Not clear if we need to replace anything but maybe root and path
	 have different dir. separators so make them all the same.  */
      std::replace (result.begin (), result.end (), '\\', '/');
      if (access (result.c_str (), F_OK) == 0)
	{
	  pdb_cvt_printf ("MSYS2_ROOT convert: '%s' -> '%s'", path,
			  result.c_str ());
	  return result;
	}
    }

  /* Fallback (a): Drive-letter mount /c/foo -> C:/foo.  */
  if (isalpha ((unsigned char) path[1]) && (path[2] == '/' || path[2] == '\0'))
    {
      std::string result;
      result += (char) toupper ((unsigned char) path[1]);
      result += ':';
      result += (path[2] == '/') ? &path[2] : "/";
      if (access (result.c_str (), F_OK) == 0)
	{
	  pdb_cvt_printf ("fallback (a) drive mount: '%s' -> '%s'", path,
			  result.c_str ());
	  return result;
	}
    }

  /* Fallback (b): Try well-known MSYS2 installation roots on all
     drive letters.  */
  static constexpr std::array<std::string_view, 4> known_suffixes
    = { ":/msys64/mingw64", ":/msys64", ":/msys2/mingw64", ":/msys2" };

  for (char drive = 'A'; drive <= 'Z'; drive++)
    {
      for (const auto &suffix : known_suffixes)
	{
	  std::string full_path = std::string (1, drive) + std::string (suffix)
				  + path;
	  if (access (full_path.c_str (), F_OK) == 0)
	    {
	      pdb_cvt_printf ("fallback (b) known root: '%s' -> '%s'", path,
			      full_path.c_str ());
	      return full_path;
	    }
	}
    }

  /* Nothing resolved; return the original path unchanged.  */
  return std::string (path);
}

/* See pdb-internal.h.  */

std::string
pdb_convert_path (const char *path)
{
  /* Non-DOS hosts do not treat '\' as a separator; Windows takes both.  */
  std::string result = pdb_resolve_path (path);
  std::replace (result.begin (), result.end (), '\\', '/');
  return result;
}

/* Return whether PATH exists.  */

static bool
file_present (const std::string &path, const char *msg)
{
  if (!path.empty () && access (path.c_str (), F_OK) == 0)
    {
      pdb_dbg_printf ("%s: %s", msg, path.c_str ());
      return true;
    }
  return false;
}

/* Return the directory portion of PATH (up to and including the last
   separator).  Returns a view into the input.  */

static std::string_view
pdb_get_dirname (std::string_view path)
{
  size_t pos = path.find_last_of ("/\\");
  if (pos == std::string_view::npos)
    return {};
  return path.substr (0, pos + 1);
}

/* Extract the basename portion of PATH (after last / or \).
   Returns a view into the input.  */

enum class strip_ext
{
  no,
  yes
};

static std::string_view
pdb_get_basename (std::string_view path, strip_ext strip = strip_ext::no)
{
  size_t last = path.find_last_of ("/\\");
  if (last != std::string_view::npos)
    path.remove_prefix (last + 1);

  if (strip == strip_ext::yes)
    if (size_t dot = path.find_last_of ('.'); dot != std::string_view::npos)
      path.remove_suffix (path.size () - dot);

  return path;
}

/* Find the section containing the virtual address range [VA, VA + SIZE).  */

static asection *
find_section_for_vma (bfd *abfd, bfd_vma va, bfd_size_type size)
{
  for (asection *sect : gdb_bfd_sections (abfd))
    {
      bfd_vma sect_va = bfd_section_vma (sect);
      bfd_size_type sect_sz = bfd_section_size (sect);
      if (va >= sect_va && va + size <= sect_va + sect_sz)
	return sect;
    }

  return nullptr;
}

/* Find the PDB path in the PE optional header debug data directory.
   The PE optional header has DataDirectory[PE_DEBUG_DATA] which gives the RVA
   and size of an array of IMAGE_DEBUG_DIRECTORY entries.  We locate the section
   containing that RVA, read the directory entries, and look for one with
   Type == PE_IMAGE_DEBUG_TYPE_CODEVIEW.  Its AddressOfRawData is an RVA
   pointing to a CV_INFO_PDB70 record (older formats are not supported).
   References:
     https://learn.microsoft.com/en-us/windows/win32/debug/pe-format
     https://github.com/Microsoft/microsoft-pdb/blob/master/include/cvinfo.h
     https://github.com/Microsoft/microsoft-pdb/blob/master/PDB/dbi/locator.cpp

   Returns RSDS info including GUID, Age, PDB path.  */

std::optional<pdb_rsds_info>
pdb_read_rsds_info (bfd *abfd, const objfile *objfile)
{
  /* pe_data() gives us the parsed PE optional header.  pe_opthdr is an
     embedded struct (not a pointer) so its address is always valid.  If no
     optional header was present the fields are zero-initialized.  */
  const internal_extra_pe_aouthdr *pe_hdr = &pe_data (abfd)->pe_opthdr;

  bfd_vma image_base = pe_hdr->ImageBase;
  bfd_vma dbg_dir_rva = pe_hdr->DataDirectory[PE_DEBUG_DATA].VirtualAddress;
  bfd_size_type dbg_dir_sz = pe_hdr->DataDirectory[PE_DEBUG_DATA].Size;

  if (dbg_dir_rva == 0 || dbg_dir_sz == 0)
    return std::nullopt;

  /* DataDirectory stores RVAs; bfd_section_vma() returns image_base + RVA.
     Convert to VA before locating the containing section.  */
  bfd_vma dbg_dir_va = dbg_dir_rva + image_base;
  asection *dbg_dir_sect = find_section_for_vma (abfd, dbg_dir_va, dbg_dir_sz);
  if (dbg_dir_sect == nullptr)
    return std::nullopt;

  /* Read the debug directory contents from its section.  */
  bfd_vma dir_offset = dbg_dir_va - bfd_section_vma (dbg_dir_sect);
  auto dir_data = std::make_unique<gdb_byte[]> (dbg_dir_sz);
  if (!bfd_get_section_contents (abfd, dbg_dir_sect, dir_data.get (),
				 dir_offset, dbg_dir_sz))
    return std::nullopt;

  /* Walk debug directory entries.  The IMAGE_DEBUG_DIRECTORY structure
     is fixed-layout (28 bytes per entry) per PE spec.  We read fields
     using fixed byte offsets to avoid depending on struct definitions
     that require machine-specific includes.  */
  /* sizeof (IMAGE_DEBUG_DIRECTORY) per PE spec.  */
  constexpr size_t entry_sz = 28;
  constexpr size_t DEBUG_DIR_TYPE_OFFS = 12;
  constexpr size_t DEBUG_DIR_SIZE_OFFS = 16;
  constexpr size_t DEBUG_DIR_ADDR_OFFS = 20;

  /* CV_INFO_PDB70 structure offsets (fixed per CodeView spec).  */
  constexpr size_t CV_PDB70_CVSIG_OFFS = 0;
  constexpr size_t CV_PDB70_SIG_OFFS = 4;
  constexpr size_t CV_PDB70_AGE_OFFS = 20;
  constexpr size_t CV_PDB70_PATH_OFFS = 24;

  size_t num_entries = dbg_dir_sz / entry_sz;

  for (size_t i = 0; i < num_entries; i++)
    {
      gdb_byte *entry = &dir_data[i * entry_sz];

      uint32_t type = bfd_h_get_32 (abfd, entry + DEBUG_DIR_TYPE_OFFS);
      uint32_t cv_sz = bfd_h_get_32 (abfd, entry + DEBUG_DIR_SIZE_OFFS);
      uint32_t cv_rva = bfd_h_get_32 (abfd, entry + DEBUG_DIR_ADDR_OFFS);

      /* Need at least header + 1 byte for filename.  */
      if (type != PE_IMAGE_DEBUG_TYPE_CODEVIEW || cv_rva == 0
	  || cv_sz <= CV_PDB70_PATH_OFFS)
	continue;

      /* Find the section containing the CodeView record.  */
      bfd_vma cv_va = (bfd_vma) cv_rva + image_base;
      asection *cv_sect = find_section_for_vma (abfd, cv_va, cv_sz);
      if (cv_sect == nullptr)
	continue;

      bfd_vma cv_offset = cv_va - bfd_section_vma (cv_sect);
      auto cv_data = std::make_unique<gdb_byte[]> (cv_sz);
      if (!bfd_get_section_contents (abfd, cv_sect, cv_data.get (), cv_offset,
				     cv_sz))
	continue;

      /* Only RSDS (PDB 7.0) is supported.  Otherwise fallback.  */
      if (bfd_h_get_32 (abfd, cv_data.get () + CV_PDB70_CVSIG_OFFS)
	  != CVINFO_PDB70_CVSIGNATURE)
	{
	  pdb_dbg_printf ("CodeView entry found but not PDB 7.0 (RSDS)");
	  return std::nullopt;
	}

      pdb_rsds_info rsds;
      memcpy (rsds.guid.data (), cv_data.get () + CV_PDB70_SIG_OFFS,
	      sizeof (rsds.guid));
      rsds.age = bfd_h_get_32 (abfd, cv_data.get () + CV_PDB70_AGE_OFFS);

      bfd_size_type path_max = cv_sz - CV_PDB70_PATH_OFFS;
      auto pdb_path = (const char *) (cv_data.get () + CV_PDB70_PATH_OFFS);
      bfd_size_type path_len = strnlen (pdb_path, path_max);

      /* Confirm the RSDS actually addresses a PDB - if empty or unterminated
	 (observed with LLD & PE & DWARF) just fallback.  */
      if (path_len == 0 || path_len >= path_max)
	{
	  pdb_dbg_printf ("CodeView RSDS present but no PDB filename");
	  return std::nullopt;
	}

      rsds.pdb_path = pdb_path;
      if (objfile->flags & OBJF_MAINLINE)
	pdb_dbg_printf ("Found PDB in RSDS: %s", rsds.pdb_path.c_str ());

      return rsds;
    }

  return std::nullopt;
}

/* See pdb-internal.h.  */

std::string
pdb_find_pdb_file (const objfile *objfile, const pdb_rsds_info &rsds)
{
  pdb_dbg_printf ("RSDS info:guid=%s age=%u path=%s",
		  pdb_format_guid (rsds.guid.data ()).c_str (), rsds.age,
		  rsds.pdb_path.c_str ());

  const char *exe_path = bfd_get_filename (objfile->obfd.get ());
  std::string_view exe_dir = pdb_get_dirname (exe_path);
  std::string_view exe_name = pdb_get_basename (exe_path, strip_ext::yes);
  std::string path;

  /* Derive the PDB basename from the RSDS record when it is present.  */
  std::string pdb_path;
  std::string_view pdb_basename;

  if (!rsds.pdb_path.empty ())
    {
      pdb_path = rsds.pdb_path;
      std::replace (pdb_path.begin (), pdb_path.end (), '\\', '/');
      pdb_basename = pdb_get_basename (pdb_path);
    }

  /* Try "EXE DIR / PDB BASENAME" path.  */
  if (!pdb_basename.empty () && !exe_dir.empty ())
    {
      path = std::string (exe_dir) + std::string (pdb_basename);
      if (file_present (path, "Using PDB basename from RSDS"))
	return path;
    }

  /* Try EXE path with extension replaced by .pdb  */
  if (exe_path != nullptr && exe_path[0] != '\0')
    {
      path = std::string (exe_dir) + std::string (exe_name) + ".pdb";
      if (file_present (path, "Using PDB path next to EXE"))
	return path;
    }

  /* Try the RSDS path as-is.  */
  if (!pdb_path.empty ())
    {
      path = pdb_path;
      if (file_present (path, "Using PDB path from RSDS"))
	return path;
    }

  /* TODO: complain when the objfile is not the main file.  */
  if ((objfile->flags & OBJF_MAINLINE) && info_verbose)
    {
      if (!pdb_path.empty ())
	printf_unfiltered ("PDB file not found for RSDS: %s\n",
			   pdb_path.c_str ());
      else
	printf_unfiltered ("PDB file not found for EXE: %s\n", exe_path);
    }

  return "";
}

} /* namespace pdb */
