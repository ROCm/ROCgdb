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

   This file implements PDB file path discovery for PE/COFF executables.  */

#include "objfiles.h"
#include "pdb/pdb-internal.h"
#include "gdbsupport/rsp-low.h"
#include "gdbsupport/gdb_vecs.h"

#include "coff/internal.h"
#include "libcoff.h"

#include <string.h>
#include <vector>
#include <errno.h>
#include <sys/stat.h>
#ifdef _WIN32
#include <direct.h>
#include <stdint.h>
#endif

/* "RSDS" magic — PDB 7.0 CodeView signature.  Defined in
   <coff/pe.h>, but including it here conflicts with the COFF
   headers already included above.  */
inline constexpr auto CVINFO_PDB70_CVSIGNATURE = 0x53445352;

/* Cannot include <winsock2.h> because it conflicts with coff/internal.h.
   FIXME: Move into a separate file then.  */
#ifdef _WIN32
extern "C"
{
  /* ws2_32.dll.  */
  int __stdcall WSAStartup (unsigned short, void *);

  /* advapi32.dll — registry.  */
  long __stdcall RegOpenKeyExA (void *, const char *, unsigned long,
				unsigned long, void **);
  long __stdcall RegQueryValueExA (void *, const char *, unsigned long *,
				   unsigned long *, unsigned char *,
				   unsigned long *);
  long __stdcall RegCloseKey (void *);
  /* kernel32.dll.  */
  unsigned long __stdcall ExpandEnvironmentStringsA (const char *, char *,
						     unsigned long);
}

#define PDB_HKCU ((void *) (intptr_t) 0x80000001)
#define PDB_HKLM ((void *) (intptr_t) 0x80000002)

#endif

namespace pdb
{

/* Check if PATH refers to an existing file.  */
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

/* Fetch PDBs from symbol server.  */
static std::string
pdb_symserver_fetch (const std::string &entry, std::string_view pdb_name,
		     const pdb_rsds_info &rsds)
{
  (void) entry;
  (void) pdb_name;
  (void) rsds;
  return "Windows symserver fetch not implemented yet";
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
static std::string_view
pdb_get_basename (std::string_view path, bool no_ext = false)
{
  size_t last = path.find_last_of ("/\\");
  size_t start = (last == std::string_view::npos) ? 0 : last + 1;

  if (!no_ext)
    return path.substr (start);

  size_t dot = path.find_last_of ('.');
  if (dot == std::string_view::npos || dot < start)
    return path.substr (start);

  return path.substr (start, dot - start);
}

/* Get file extension. E.g. filename.exe -> exe.
   Returns a view into the input.  */
static std::string_view
pdb_get_extension (std::string_view path)
{
  size_t slash = path.find_last_of ("/\\");
  size_t dot = path.find_last_of ('.');

  if (dot == std::string_view::npos
      || (slash != std::string_view::npos && dot < slash))
    return {};

  return path.substr (dot + 1);
}

/* Walk a semicolon-separated search path looking for PDB_NAME.  For each
   path component PATH:
     - For [PATH=SRV|SYMSRV|CACHE]: attempt symbol server fetch.
     - For plain directories: try PATH/<ext>/PDB_NAME then PATH/PDB_NAME.
       <ext> is the file extension of EXE_PATH (e.g.  "exe", "dll").
       Returns the first path that exists, or empty string.  */
static std::string
pdb_search_path (const char *search_path, std::string_view pdb_name,
		 std::string_view exe_name, std::string_view exe_ext,
		 const pdb_rsds_info &rsds)
{
  if (search_path == nullptr || search_path[0] == '\0')
    return "";

  std::string basename = pdb_name.empty () ? std::string (exe_name) + ".pdb"
					   : std::string (pdb_name);
  if (basename.empty ())
    {
      return "";
    }

  const char *p = search_path;
  while (*p != '\0')
    {
      /* Extract one semicolon-delimited component.  */
      const char *token = strchr (p, ';');

      /* Length of the dir is either one token or the whole search path.  */
      size_t len = token ? (token - p) : strlen (p);
      /* Get the dir to analyze.  */
      std::string dir (p, len);

      p += len + (token ? 1 : 0);

      if (dir.empty ())
	continue;

      /* Handle symbol server entries.  */
      if (dir.compare (0, 4, "SRV*") == 0 || dir.compare (0, 7, "SYMSRV*") == 0
	  || dir.compare (0, 6, "CACHE*") == 0)
	{
	  std::string path = pdb_symserver_fetch (dir, pdb_name, rsds);
	  if (access (path.c_str (), F_OK) == 0)
	    return path;
	  continue;
	}

      /* make sure the directory ends with /.  */
      if (char last_ch = dir.back (); last_ch != '/' && last_ch != '\\')
	dir += '/';

      /* Try <dir>/<ext>/<pdb>.  */
      if (!exe_ext.empty ())
	{
	  std::string path = dir + std::string (exe_ext) + "/" + basename;
	  if (access (path.c_str (), F_OK) == 0)
	    return path;
	}

      /* Try <dir>/<pdb>.  */
      std::string path = dir + basename;
      if (access (path.c_str (), F_OK) == 0)
	return path;
    }

  return "";
}

#ifdef _WIN32
/* Read SymbolSearchPath from the Windows registry.  The MS PDB locator
  (see microsoft-pdb/PDB/dbi/locator.cpp) reads
   this value from both HKEY_CURRENT_USER and HKEY_LOCAL_MACHINE under
     Software\Microsoft\VisualStudio\MSPDB
   The value may be REG_SZ (1) or REG_EXPAND_SZ (2) with %VAR%
   references.  Returns the search path, or empty string.  */
static std::string
pdb_read_win_registry_search_path (void *root_key)
{
  void *hkey;
  long rc = RegOpenKeyExA (root_key,
			   "Software\\Microsoft\\VisualStudio\\MSPDB", 0,
			   /*KEY_QUERY_VALUE.  */ 0x0001, &hkey);
  if (rc != 0)
    return "";

  /* Get the required buffer size.  */
  unsigned long cb = 0;
  rc = RegQueryValueExA (hkey, "SymbolSearchPath", nullptr, nullptr, nullptr,
			 &cb);
  if (rc != 0 || cb == 0)
    {
      RegCloseKey (hkey);
      return "";
    }

  /* Allocate buffer and read the value.  */
  std::string buf (cb, '\0');
  unsigned long dwType;
  rc = RegQueryValueExA (hkey, "SymbolSearchPath", nullptr, &dwType,
			 (unsigned char *) &buf[0], &cb);
  RegCloseKey (hkey);

  if (rc != 0)
    return "";

  /* Strip trailing NUL(s) that RegQueryValueExA might include.  */
  while (!buf.empty () && buf.back () == '\0')
    buf.pop_back ();

  if (buf.empty ())
    return "";

  if (dwType == 1 /* REG_SZ.  */)
    return buf;

  if (dwType == 2 /* REG_EXPAND_SZ.  */)
    {
      /* Expand environment variables like %SYSTEMROOT%.  */
      unsigned long needed = ExpandEnvironmentStringsA (buf.c_str (), nullptr,
							0);
      if (needed == 0)
	return "";
      std::string expanded (needed, '\0');
      unsigned long written = ExpandEnvironmentStringsA (buf.c_str (),
							 &expanded[0], needed);
      if (written == 0)
	return "";
      /* Strip trailing NUL.  */
      while (!expanded.empty () && expanded.back () == '\0')
	expanded.pop_back ();
      return expanded;
    }

  /* Unexpected registry type — ignore.  */
  return "";
}

#endif /* _WIN32.  */

/* Find the section containing virtual address VA.
   TODO: Is there already a bfd function for this ?  */
static asection *
find_section_for_vma (bfd *abfd, bfd_vma va, bfd_size_type size)
{
  for (asection *sect = abfd->sections; sect != nullptr; sect = sect->next)
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
  const size_t entry_sz = 28; /* sizeof (IMAGE_DEBUG_DIRECTORY) per PE spec */
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
      gdb_byte *entry = dir_data.get () + (i * entry_sz);

      uint32_t type = H_GET_32 (abfd, (void *) (entry + DEBUG_DIR_TYPE_OFFS));
      uint32_t cv_sz = H_GET_32 (abfd, (void *) (entry + DEBUG_DIR_SIZE_OFFS));
      uint32_t cv_rva = H_GET_32 (abfd,
				  (void *) (entry + DEBUG_DIR_ADDR_OFFS));

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

      /* Only RSDS (PDB 7.0) is supported.  */
      if (H_GET_32 (abfd, (void *) (cv_data.get () + CV_PDB70_CVSIG_OFFS))
	  != CVINFO_PDB70_CVSIGNATURE)
	{
	  pdb_warning ("CodeView entry found but not PDB 7.0 (RSDS)");
	  /* Return an empty rsds — the binary owns CodeView so DWARF
	     must not be invoked, but we can't extract a path.  */
	  return pdb_rsds_info {};
	}

      pdb_rsds_info rsds;
      memcpy (rsds.guid.data (), cv_data.get () + CV_PDB70_SIG_OFFS,
	      sizeof (rsds.guid));
      rsds.age = H_GET_32 (abfd,
			   (void *) (cv_data.get () + CV_PDB70_AGE_OFFS));

      bfd_size_type path_max = cv_sz - CV_PDB70_PATH_OFFS;
      auto pdb_path = (const char *) (cv_data.get () + CV_PDB70_PATH_OFFS);
      if (bfd_size_type path_len = strnlen (pdb_path, path_max);
	  path_len > 0 && path_len < path_max)
	{
	  rsds.pdb_path = pdb_path;
	  if (objfile->flags & OBJF_MAINLINE)
	    pdb_dbg_printf ("Found PDB in RSDS: %s", rsds.pdb_path.c_str ());
	}

      return rsds;
    }

  return std::nullopt;
}

/* See pdb.h.  */
std::string
pdb_find_pdb_file (const objfile *objfile, const pdb_rsds_info &rsds)
{
  pdb_dbg_printf ("RSDS info:guid=%s age=%u path=%s",
		  pdb_format_guid (rsds.guid.data ()).c_str (), rsds.age,
		  rsds.pdb_path.c_str ());

  const char *exe_path = bfd_get_filename (objfile->obfd.get ());
  std::string_view exe_dir = pdb_get_dirname (exe_path);
  std::string_view exe_name = pdb_get_basename (exe_path, true);
  std::string_view exe_ext = pdb_get_extension (exe_path);
  std::string path;

  /* Take the basename from RSDS record, if it doesn't exist have pdb basename
     as EXE name + pdb.  */
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

  /* Try each entry in debug-file-directory.  */
  if (!debug_file_directory.empty () && !pdb_basename.empty ())
    {
      std::vector<gdb::unique_xmalloc_ptr<char>> dirs
	= dirnames_to_char_ptr_vec (debug_file_directory.c_str ());
      for (const auto &dir : dirs)
	{
	  path = std::string (dir.get ()) + "/" + std::string (pdb_basename);
	  if (file_present (path, "Using PDB from debug-file-directory"))
	    return path;
	}
    }

  /* Try as is RSDS name.  */
  if (!pdb_path.empty ())
    {
      path = pdb_path;
      if (file_present (path, "Using PDB path from RSDS"))
	return path;
    }

  /* Walk _NT_ALT_SYMBOL_PATH and _NT_SYMBOL_PATH env. vars.
     For each <path> try <path>/<ext>/<pdb> then <path>/<pdb>.
     See https://learn.microsoft.com/en-us/windows/win32/debug/symbol-paths
     and microsoft-pdb locator.cpp.  */
  const char *env = getenv ("_NT_ALT_SYMBOL_PATH");
  path = pdb_search_path (env, pdb_basename, exe_name, exe_ext, rsds);
  if (file_present (path, "Using PDB path from _NT_ALT_SYMBOL_PATH"))
    return path;

  env = getenv ("_NT_SYMBOL_PATH");
  path = pdb_search_path (env, pdb_basename, exe_name, exe_ext, rsds);
  if (file_present (path, "Using PDB path from _NT_SYMBOL_PATH"))
    return path;

  /* Windows registry — walk SymbolSearchPath from
     HKCU then HKLM under VisualStudio\MSPDB.  */
#ifdef _WIN32
  std::string reg = pdb_read_win_registry_search_path (PDB_HKCU);
  path = pdb_search_path (reg.c_str (), pdb_basename, exe_name, exe_ext, rsds);
  if (file_present (path, "Using PDB path from HKCU"))
    return path;

  reg = pdb_read_win_registry_search_path (PDB_HKLM);
  path = pdb_search_path (reg.c_str (), pdb_basename, exe_name, exe_ext, rsds);
  if (file_present (path, "Using PDB path from HKLM"))
    return path;

#endif /* _WIN32.  */
  /* TODO - complaint if not mainfile */
  if (objfile->flags & OBJF_MAINLINE)
    {
      if (!pdb_path.empty ())
	printf_unfiltered ("PDB file not found for RSDS: %s\n",
			   pdb_path.c_str ());
      else
	printf_unfiltered ("PDB file not found for EXE: %s\n", exe_path);
    }

  return "";
}

} // namespace pdb
