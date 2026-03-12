/* PDB debugging format support for GDB.

   Copyright (C) 1994-2026 Free Software Foundation, Inc.
   Copyright (C) 1994-2026 Advanced Micro Devices, Inc. All rights reserved.

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
#include "pdb/pdb.h"
#include "gdbsupport/rsp-low.h"

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

/* Return and print the message if the file is present.  */
#define RET_IF_FILE_PRESENT(p, msg)                    \
  do {                                                 \
    if (!p.empty() && access(p.c_str(), F_OK) == 0) {  \
      pdb_dbg_printf("%s: %s", (msg), p.c_str());      \
      return p;                                        \
    }                                                  \
  } while (0)

/* Cannot include <winsock2.h> because it conflicts with coff/internal.h.
   FIXME: Move into a separate file then.  */
#ifdef _WIN32
extern "C" {
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

#define PDB_HKCU  ((void *)(intptr_t)0x80000001)
#define PDB_HKLM  ((void *)(intptr_t)0x80000002)

#endif

/* See pdb.h.  */
const char *pdb_file_override = NULL;

/* Fetch PDBs from symbol server.  */
std::string
pdb_symserver_fetch (const std::string &entry, const std::string &pdb_name,
		     const struct pdb_rsds_info &rsds)
{
  (void) entry;
  (void) pdb_name;
  (void) rsds;
  return "Windows symserver fetch not implemented yet";
}

/* See pdb.h.  */
static std::string
pdb_get_dirname (const std::string &path)
{
  size_t pos = path.find_last_of ("/\\");
  if (pos == std::string::npos)
    return "";
  return path.substr (0, pos + 1);
}

/* Extract the basename portion of PATH (after last / or \).  */
static std::string
pdb_get_basename(const std::string &path, bool no_ext = false)
{
  size_t last = path.find_last_of("/\\");
  size_t start = (last == std::string::npos) ? 0 : last + 1;

  if (!no_ext)
    return path.substr(start);

  size_t dot = path.find_last_of('.');
  if (dot == std::string::npos || dot < start)
    return path.substr(start);

  return path.substr(start, dot - start);
}

/* Get file extension. E.g. filename.exe -> exe  */
static std::string pdb_get_extension(const std::string& path)
{
  size_t slash = path.find_last_of("/\\");
  size_t dot   = path.find_last_of('.');

  if (dot == std::string::npos || (slash != std::string::npos && dot < slash))
    return "";

  return path.substr(dot + 1);
}

/* Walk a semicolon-separated search path looking for PDB_NAME.  For each
   path component PATH:
     - For [PATH=SRV|SYMSRV|CACHE]: attempt symbol server fetch.
     - For plain directories: try PATH/<ext>/PDB_NAME then PATH/PDB_NAME.
       <ext> is the file extension of EXE_PATH (e.g.  "exe", "dll").
       Returns the first path that exists, or empty string.  */
static std::string
pdb_search_path (const char *search_path, const std::string &pdb_name,
		 const std::string &exe_name, const std::string &exe_ext,
		 const struct pdb_rsds_info &rsds)
{
  if (search_path == NULL || search_path[0] == '\0')
    return "";

  std::string basename = pdb_name.empty() ? exe_name + ".pdb" : pdb_name;
  if (basename.empty())
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
      if (dir.compare (0, 4, "SRV*") == 0
	  || dir.compare (0, 7, "SYMSRV*") == 0
	  || dir.compare (0, 6, "CACHE*") == 0)
	{
	  std::string path = pdb_symserver_fetch (dir, pdb_name, rsds);
	  if (access(path.c_str(), F_OK) == 0)
	    return path;
	  continue;
	}

      /* make sure the directory ends with /.  */
      char last_ch = dir.back ();
      if (last_ch != '/' && last_ch != '\\')
	dir += '/';

      /* Try <dir>/<ext>/<pdb>.  */
      if (!exe_ext.empty ())
	{
	  std::string path = dir + exe_ext + "/" + basename;
	  if (access(path.c_str(), F_OK) == 0)
	    return path;
	}

      /* Try <dir>/<pdb>.  */
      std::string path = dir + basename;
      if (access(path.c_str(), F_OK) == 0)
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
			   "Software\\Microsoft\\VisualStudio\\MSPDB",
			   0, /*KEY_QUERY_VALUE.  */ 0x0001, &hkey);
  if (rc != 0)
    return "";

  /* Get the required buffer size.  */
  unsigned long cb = 0;
  rc = RegQueryValueExA (hkey, "SymbolSearchPath", NULL, NULL, NULL, &cb);
  if (rc != 0 || cb == 0)
    {
      RegCloseKey (hkey);
      return "";
    }

  /* Allocate buffer and read the value.  */
  std::string buf (cb, '\0');
  unsigned long dwType;
  rc = RegQueryValueExA (hkey, "SymbolSearchPath", NULL, &dwType,
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
      unsigned long needed = ExpandEnvironmentStringsA (buf.c_str (), NULL, 0);
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
  for (asection *sect = abfd->sections; sect != NULL; sect = sect->next)
    {
      bfd_vma sect_va = bfd_section_vma (sect);
      bfd_size_type sect_sz = bfd_section_size (sect);
      if (va >= sect_va && va + size <= sect_va + sect_sz)
	return sect;
    }
  return NULL;
}

/* Find the PDB path in the PE optional header debug data directory.
   The PE optional header has DataDirectory[PE_DEBUG_DATA] which gives the RVA
   and size of an array of IMAGE_DEBUG_DIRECTORY entries.  We locate the section
   containing that RVA, read the directory entries, and look for one with
   DebugType == CODEVIEW (2). That entry's AddressOfRawData is an RVA pointing
   to the CodeView PDB70 record with the layout:
     offset  0: CVSignature "RSDS"  (4 bytes)  = 0x53445352
     offset  4: GUID                (16 bytes)
     offset 20: Age                 (4 bytes)
     offset 24: PdbFileName         (null-terminated string)
  https://learn.microsoft.com/en-us/windows/win32/debug/pe-format
  https://github.com/Microsoft/microsoft-pdb/blob/master/include/cvinfo.h
  https://github.com/Microsoft/microsoft-pdb/blob/master/PDB/dbi/locator.cpp
   TODO: introduce some defines.

   Returns RSDS info including GUID, Age, PDB path  */
static struct pdb_rsds_info ATTRIBUTE_UNUSED
pdb_read_rsds_info (bfd *abfd, struct objfile *objfile)
{
  struct pdb_rsds_info rsds = {};
  rsds.valid = false;
  /* pe_data() gives us the parsed PE optional header.  */
  struct internal_extra_pe_aouthdr *pe_hdr = &pe_data (abfd)->pe_opthdr;
  if (pe_hdr == NULL)
    return rsds;

  bfd_vma image_base = pe_hdr->ImageBase;

  /* DataDirectory stores RVAs (relative to image base), but bfd_section_vma()
     returns full VAs (image base + RVA).  Convert RVAs to VAs by adding
     image_base before comparing.  */

  /* DataDirectory[6] is PE_DEBUG_DATA.  */
  bfd_vma dbg_dir_rva = pe_hdr->DataDirectory[PE_DEBUG_DATA].VirtualAddress;
  long dbg_dir_sz = pe_hdr->DataDirectory[PE_DEBUG_DATA].Size;
  bfd_vma dbg_dir_va  = dbg_dir_rva + image_base;

  if (dbg_dir_rva == 0 || dbg_dir_sz <= 0)
    return rsds;

  /* Find which section contains the debug directory RVA.  */
  asection *dbg_dir_sect = find_section_for_vma (abfd, dbg_dir_va, dbg_dir_sz);

  if (dbg_dir_sect == NULL)
    return rsds;

  /* Offset within the section where the directory array starts.  */
  bfd_vma dir_offset = dbg_dir_va - bfd_section_vma (dbg_dir_sect);

  /* Read the debug directory entries.  */
  std::unique_ptr<gdb_byte[]> dir_data(new gdb_byte[dbg_dir_sz]);
  if (!bfd_get_section_contents (abfd, dbg_dir_sect,
				 dir_data.get(), dir_offset, dbg_dir_sz))
    {
      return rsds;
    }

  /* IMAGE_DEBUG_DIRECTORY entry is 28 bytes:
       0: Characteristics   (4)
       4: TimeDateStamp     (4)
       8: MajorVersion      (2)
      10: MinorVersion      (2)
      12: Type              (4)
      16: SizeOfData        (4)
      20: AddressOfRawData  (4)  -- RVA of the payload
      24: PointerToRawData  (4).  */
  const int entry_sz = 28;
  int num_entries = dbg_dir_sz / entry_sz;

  for (int i = 0; i < num_entries; i++)
    {
      const gdb_byte *ent = dir_data.get() + i * entry_sz;
      uint32_t type    = bfd_get_32 (abfd, ent + 12);
      uint32_t cv_sz   = bfd_get_32 (abfd, ent + 16);
      uint32_t cv_rva  = bfd_get_32 (abfd, ent + 20);
      bfd_vma  cv_va   = (bfd_vma) cv_rva + image_base;

      /* Skip if RV is zero, type is not Debug or size is too small.
	 TODO: introduce some defines.  */
      if (type != PE_IMAGE_DEBUG_TYPE_CODEVIEW || cv_rva == 0 || cv_sz < 25)
	continue;

      /* Find the section containing the CodeView record.  */
      asection *cv_sect = find_section_for_vma (abfd, cv_va, cv_sz);
      if (cv_sect == NULL)
	continue;

      bfd_vma cv_offset = cv_va - bfd_section_vma (cv_sect);
      std::unique_ptr<gdb_byte[]> cv_data(new gdb_byte[cv_sz]);
      if (!bfd_get_section_contents (abfd, cv_sect, cv_data.get(),
				     cv_offset, cv_sz))
	{
	  continue;
	}

      /* Check for PDB 7.0 signature. It's at location zero.  */
      uint32_t cv_sig = bfd_get_32 (abfd, cv_data.get());
      if (cv_sig != 0x53445352)
	{
	  pdb_error("RSDS: Expecting PDB 7.0 signature");
	  return rsds;
	}

      /* PDB70: [sig 4] [GUID 16] [Age 4] [PdbFileName ...].  */
      memcpy (rsds.guid, cv_data.get() + 4, 16);
      rsds.age = bfd_get_32 (abfd, cv_data.get() + 20);

      const char *pdb_path = (const char *)(cv_data.get() + 24);
      bfd_size_type max_len = cv_sz - 24;
      bfd_size_type path_len = strnlen (pdb_path, max_len);
      if (path_len < max_len && path_len > 0)
	{
	  rsds.pdb_path = pdb_path;
	  rsds.valid = true;
	  /* TODO - complaint if not mainfile */
	  if (objfile->flags & OBJF_MAINLINE)
	    pdb_dbg_printf ("Found PDB in RSDS: %s", rsds.pdb_path.c_str ());
	}

      break;
    }

  return rsds;
}

/* See pdb.h.  */
std::string
pdb_find_pdb_file (struct objfile *objfile)
{
  /* Command-line override (applies to main executable only).  */
  if (pdb_file_override != NULL && (objfile->flags & OBJF_MAINLINE))
    {
      pdb_dbg_printf ("Using PDB file from command line: %s",
		       pdb_file_override);
      /* No fallback - return without testing for file presence. */
      return std::string (pdb_file_override);
    }

  /* Read the RSDS record from the PE debug directory.  */
  struct pdb_rsds_info rsds = pdb_read_rsds_info (objfile->obfd.get (), objfile);

  pdb_dbg_printf ("RSDS info:guid=%s age=%u path=%s",
		  bin2hex (rsds.guid, 16).c_str (),
		  rsds.age,
		  rsds.pdb_path.c_str ());

  std::string exe_path = std::string(bfd_get_filename (objfile->obfd.get ()));
  std::string exe_dir = pdb_get_dirname(exe_path);
  std::string exe_basename = pdb_get_basename(exe_path);
  std::string exe_name = pdb_get_basename(exe_path, true);
  std::string exe_ext = pdb_get_extension(exe_path);
  std::string path;

  /* Take the basename from RSDS record, if it doesn't exist have pdb basename
     as EXE name + pdb.  */
  std::string pdb_path;
  std::string pdb_basename;

  if (rsds.valid && !rsds.pdb_path.empty ())
    {
      pdb_path = rsds.pdb_path;
      std::replace(pdb_path.begin(), pdb_path.end(), '\\', '/');
      pdb_basename = pdb_get_basename(pdb_path.c_str());
   }

  /* Try "EXE DIR / PDB BASENAME" path.  */
  if (!pdb_basename.empty () && !exe_dir.empty ())
    {
      path = exe_dir + pdb_basename;
      RET_IF_FILE_PRESENT(path, "Using PDB basename from RSDS");
    }

  /* Try EXE path with extension replaced by .pdb  */
  if (!exe_path.empty ())
    {
      path = exe_dir + exe_name + ".pdb";
      RET_IF_FILE_PRESENT(path, "Using PDB path next to EXE");
    }

   /* Try as is RSDS name.  */
   if (!pdb_path.empty ())
    {
      path = pdb_path;
      RET_IF_FILE_PRESENT(path, "Using PDB path from RSDS");
    }

  /* Walk _NT_ALT_SYMBOL_PATH and _NT_SYMBOL_PATH env. vars.
     For each <path> try <path>/<ext>/<pdb> then <path>/<pdb>.
     See https://learn.microsoft.com/en-us/windows/win32/debug/symbol-paths
     and microsoft-pdb locator.cpp.  */
  const char *env = getenv ("_NT_ALT_SYMBOL_PATH");
  path = pdb_search_path (env, pdb_basename, exe_name, exe_ext, rsds);
  RET_IF_FILE_PRESENT(path, "Using PDB path from _NT_ALT_SYMBOL_PATH");

  env = getenv ("_NT_SYMBOL_PATH");
  path = pdb_search_path (env, pdb_basename, exe_name, exe_ext, rsds);
  RET_IF_FILE_PRESENT(path, "Using PDB path from _NT_SYMBOL_PATH");

  /* Windows registry — walk SymbolSearchPath from
     HKCU then HKLM under VisualStudio\MSPDB.  */
#ifdef _WIN32
  std::string reg = pdb_read_win_registry_search_path (PDB_HKCU);
  path = pdb_search_path (reg.c_str (), pdb_basename, exe_name, exe_ext, rsds);
  RET_IF_FILE_PRESENT(path, "Using PDB path from HKCU");

  reg = pdb_read_win_registry_search_path (PDB_HKLM);
  path = pdb_search_path (reg.c_str (), pdb_basename, exe_name, exe_ext, rsds);
  RET_IF_FILE_PRESENT(path, "Using PDB path from HKLM");

#endif /* _WIN32.  */
  /* TODO - complaint if not mainfile */
  if (objfile->flags & OBJF_MAINLINE)
  {
      if (!pdb_path.empty())
	printf_unfiltered ("PDB file not found for RSDS: %s", pdb_path.c_str());
      else
	printf_unfiltered ("PDB file not found for EXE: %s", exe_path.c_str ());
   }
  return "";
}
