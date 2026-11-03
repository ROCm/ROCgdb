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
#include "gdb_bfd.h"

#include "coff/internal.h"
#include "libcoff.h"

#include <string.h>
#include <algorithm>
#include <array>
#include <set>
#include "filenames.h"
#include "gdbsupport/selftest.h"
#include <sys/stat.h>

namespace pdb
{

/* Return whether PATH is a regular file.  A FIFO or a device by the PDB's
   name is passed over: opening one could block.  A path that cannot be
   examined is not a file either, so the search goes on to the next one.  */

static bool
file_present (const std::string &path)
{
  struct stat st;
  return stat (path.c_str (), &st) == 0 && S_ISREG (st.st_mode);
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

enum class strip_ext
{
  no,
  yes
};

/* Extract the basename portion of PATH (after last / or \), without its
   extension when STRIP is strip_ext::yes.  Returns a view into the input.  */

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

/* The index in IMAGE's sections of the one containing the virtual address
   range [VA, VA + SIZE), or -1.  */

static int
find_section_for_vma (const pdb_pe_image &image, uint64_t va, uint64_t size)
{
  for (size_t i = 0; i < image.sections.size (); i++)
    {
      /* VA and SIZE come from the file: compare without sums that can
	 wrap.  */
      const pdb_pe_image::section &sect = image.sections[i];
      if (va >= sect.vma && va - sect.vma <= sect.size
	  && size <= sect.size - (va - sect.vma))
	return i;
    }

  return -1;
}

/* See pdb-internal.h.  The PE optional header's DataDirectory[PE_DEBUG_DATA]
   gives the RVA and size of an array of IMAGE_DEBUG_DIRECTORY entries.  We
   locate the section containing that RVA, read the directory entries, and
   look for one with Type == PE_IMAGE_DEBUG_TYPE_CODEVIEW.  Its
   AddressOfRawData is an RVA pointing to a CV_INFO_PDB70 record (older
   formats are not supported).
   References:
     https://learn.microsoft.com/en-us/windows/win32/debug/pe-format
     https://github.com/Microsoft/microsoft-pdb/blob/master/include/cvinfo.h
     microsoft-pdb's PDB/dbi/locator.cpp.  */

std::optional<pdb_rsds_info>
pdb_parse_rsds (const pdb_pe_image &image)
{
  uint64_t dbg_dir_sz = image.debug_dir_size;
  if (image.debug_dir_rva == 0 || dbg_dir_sz == 0)
    return std::nullopt;

  /* The sizes come from the image's headers; neither can exceed the file.
     A size of 0 means the file size is unknown.  */
  uint64_t file_size = image.file_size;
  if (file_size != 0 && dbg_dir_sz > file_size)
    return std::nullopt;

  /* A section's address is image_base + RVA.  Convert the RVA to that
     before locating the containing section.  */
  uint64_t dbg_dir_va = image.debug_dir_rva + image.image_base;
  int dbg_dir_sect = find_section_for_vma (image, dbg_dir_va, dbg_dir_sz);
  if (dbg_dir_sect < 0)
    return std::nullopt;

  /* find_section_for_vma checked that the directory lies within the
     section's contents.  Its entries are read one at a time, so a
     large size in the file allocates nothing.  */
  uint64_t dir_offset = dbg_dir_va - image.sections[dbg_dir_sect].vma;

  /* Walk debug directory entries.  The IMAGE_DEBUG_DIRECTORY structure
     is fixed-layout (28 bytes per entry) per PE spec.  We read fields
     using fixed byte offsets to avoid depending on struct definitions
     that require machine-specific includes.  */
  constexpr size_t entry_sz = 28;
  constexpr size_t DEBUG_DIR_TYPE_OFFS = 12;
  constexpr size_t DEBUG_DIR_SIZE_OFFS = 16;
  constexpr size_t DEBUG_DIR_ADDR_OFFS = 20;
  constexpr size_t DEBUG_DIR_PTR_OFFS = 24;

  /* CV_INFO_PDB70 structure offsets (fixed per CodeView spec).  */
  constexpr size_t CV_PDB70_CVSIG_OFFS = 0;
  constexpr size_t CV_PDB70_SIG_OFFS = 4;
  constexpr size_t CV_PDB70_AGE_OFFS = 20;
  constexpr size_t CV_PDB70_PATH_OFFS = 24;

  /* The longest CV_INFO_PDB70 read: its path is a Windows path, at most
     32767 characters.  */
  constexpr size_t CV_PDB70_MAX_SIZE = CV_PDB70_PATH_OFFS + 32768;

  /* "RSDS" magic: PDB 7.0 CodeView signature.  coff/pe.h defines it as the
     macro CVINFO_PDB70_CVSIGNATURE, but including that here conflicts with
     the COFF headers already included above.  */
  constexpr uint32_t rsds_signature = 0x53445352;

  size_t num_entries = dbg_dir_sz / entry_sz;

  for (size_t i = 0; i < num_entries; i++)
    {
      gdb_byte entry[entry_sz];
      if (!image.read (dbg_dir_sect, entry, dir_offset + i * entry_sz,
		       entry_sz))
	return std::nullopt;

      uint32_t type = read_u32 (entry + DEBUG_DIR_TYPE_OFFS);
      uint32_t cv_sz = read_u32 (entry + DEBUG_DIR_SIZE_OFFS);
      uint32_t cv_rva = read_u32 (entry + DEBUG_DIR_ADDR_OFFS);
      uint32_t cv_ptr = read_u32 (entry + DEBUG_DIR_PTR_OFFS);

      /* Need at least header + 1 byte for filename.  */
      if (type != PE_IMAGE_DEBUG_TYPE_CODEVIEW
	  || cv_sz <= CV_PDB70_PATH_OFFS || cv_sz > CV_PDB70_MAX_SIZE
	  || (file_size != 0 && cv_sz > file_size))
	continue;

      /* Read the CodeView record from the section that contains it.  An
	 entry that is not mapped has no address (AddressOfRawData is 0), as
	 BFD notes: its record is then read at PointerToRawData, in the
	 file.  */
      auto cv_data = std::make_unique<gdb_byte[]> (cv_sz);
      bool have_cv = false;
      if (cv_rva != 0)
	{
	  uint64_t cv_va = (uint64_t) cv_rva + image.image_base;
	  int cv_sect = find_section_for_vma (image, cv_va, cv_sz);
	  if (cv_sect >= 0)
	    have_cv = image.read (cv_sect, cv_data.get (),
				  cv_va - image.sections[cv_sect].vma, cv_sz);
	}
      if (!have_cv && cv_ptr != 0
	  && (file_size == 0 || (uint64_t) cv_ptr + cv_sz <= file_size))
	have_cv = image.read_file (cv_ptr, cv_data.get (), cv_sz);
      if (!have_cv)
	continue;

      /* Only RSDS (PDB 7.0) is supported.  Otherwise fall back to other
	 debug information.  */
      if (read_u32 (cv_data.get () + CV_PDB70_CVSIG_OFFS)
	  != rsds_signature)
	{
	  pdb_dbg_printf ("CodeView entry found but not PDB 7.0 (RSDS)");
	  return std::nullopt;
	}

      pdb_rsds_info rsds;
      memcpy (rsds.guid.data (), cv_data.get () + CV_PDB70_SIG_OFFS,
	      sizeof (rsds.guid));
      rsds.age = read_u32 (cv_data.get () + CV_PDB70_AGE_OFFS);

      size_t path_max = cv_sz - CV_PDB70_PATH_OFFS;
      auto pdb_path = (const char *) (cv_data.get () + CV_PDB70_PATH_OFFS);
      size_t path_len = strnlen (pdb_path, path_max);

      /* Confirm the RSDS actually addresses a PDB - if empty or unterminated
	 (observed with LLD & PE & DWARF) just fall back.  */
      if (path_len == 0 || path_len >= path_max)
	{
	  pdb_dbg_printf ("CodeView RSDS present but no PDB filename");
	  return std::nullopt;
	}

      rsds.pdb_path = pdb_path;
      return rsds;
    }

  return std::nullopt;
}

/* Find the PDB path in the PE optional header debug data directory of
   ABFD.  Returns RSDS info including GUID, Age, PDB path.  */

std::optional<pdb_rsds_info>
pdb_read_rsds_info (bfd *abfd)
{
  /* Only a PE file has the PE data read below; other COFF files, such
     as go32 executables, have no debug directory.  */
  const char *target = bfd_get_target (abfd);
  if (!startswith (target, "pe") && !startswith (target, "epoc-pe"))
    return std::nullopt;

  /* pe_data() gives us the parsed PE optional header.  pe_opthdr is an
     embedded struct (not a pointer) so its address is always valid.  If no
     optional header was present the fields are zero-initialized.  */
  const internal_extra_pe_aouthdr *pe_hdr = &pe_data (abfd)->pe_opthdr;

  pdb_pe_image image;
  image.image_base = pe_hdr->ImageBase;
  image.debug_dir_rva = pe_hdr->DataDirectory[PE_DEBUG_DATA].VirtualAddress;
  image.debug_dir_size = pe_hdr->DataDirectory[PE_DEBUG_DATA].Size;
  image.file_size = bfd_get_file_size (abfd);

  /* bfd_section_vma() returns image_base + RVA.  */
  std::vector<asection *> sects;
  for (asection *sect : gdb_bfd_sections (abfd))
    {
      image.sections.push_back ({ bfd_section_vma (sect),
				  bfd_section_size (sect) });
      sects.push_back (sect);
    }
  auto read = [&] (size_t i, gdb_byte *buf, uint64_t offset, uint64_t len)
    {
      return bfd_get_section_contents (abfd, sects[i], buf, offset, len);
    };
  image.read = read;
  auto read_file = [&] (uint64_t offset, gdb_byte *buf, uint64_t len)
    {
      return (bfd_seek (abfd, offset, SEEK_SET) == 0
	      && bfd_read (buf, len, abfd) == len);
    };
  image.read_file = read_file;

  std::optional<pdb_rsds_info> rsds = pdb_parse_rsds (image);
  if (rsds.has_value ())
    pdb_dbg_printf ("Found PDB in RSDS: %s", rsds->pdb_path.c_str ());
  return rsds;
}

/* See pdb-internal.h.  */

std::string
pdb_find_pdb_file (const objfile *objfile, const pdb_rsds_info &rsds)
{
  return pdb_find_pdb_file (bfd_get_filename (objfile->obfd.get ()), rsds,
			    (objfile->flags & OBJF_MAINLINE) != 0,
			    file_present);
}

/* See pdb-internal.h.  */

std::string
pdb_find_pdb_file (const char *exe_path, const pdb_rsds_info &rsds,
		   bool mainline, pdb_file_present_ftype present)
{
  pdb_dbg_printf ("RSDS info: guid=%s age=%u path=%s",
		  pdb_format_guid (rsds.guid.data ()).c_str (), rsds.age,
		  rsds.pdb_path.c_str ());

  /* Whether PATH is a file, reported with how it was found, MSG.  */
  auto found = [&] (const std::string &path, const char *msg)
    {
      if (path.empty () || !present (path))
	return false;
      pdb_dbg_printf ("%s: %s", msg, path.c_str ());
      return true;
    };

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
  if (!pdb_basename.empty ())
    {
      path = std::string (exe_dir) + std::string (pdb_basename);
      if (found (path, "Using PDB basename from RSDS"))
	return path;
    }

  /* Try EXE path with extension replaced by .pdb.  */
  if (exe_path[0] != '\0')
    {
      path = std::string (exe_dir) + std::string (exe_name) + ".pdb";
      if (found (path, "Using PDB path next to EXE"))
	return path;
    }

  /* Try the RSDS path as-is, unless it is a network path
     (\\host\share\...) or a device path (\\?\..., \\.\...): opening one
     named by an untrusted executable makes Windows send the user's
     credentials to that host.  Both begin with two separators, which were
     all made '/' above.  Nor is a path that is not absolute on this host
     tried, such as a Windows path on a POSIX host: it would be looked for
     in GDB's working directory.  */
  bool network_path = startswith (pdb_path, "//");
  /* A drive-relative path, "C:foo.pdb", is relative to the drive's
     current directory, so it is not absolute either.  */
  bool absolute = (IS_ABSOLUTE_PATH (pdb_path.c_str ())
		   && !(HAS_DRIVE_SPEC (pdb_path.c_str ())
			&& !IS_DIR_SEPARATOR (pdb_path[2])));
  if (network_path)
    pdb_dbg_printf ("Not opening the network PDB path from RSDS: %s",
		    pdb_path.c_str ());
  else if (!pdb_path.empty () && !absolute)
    pdb_dbg_printf ("Not opening the relative PDB path from RSDS: %s",
		    pdb_path.c_str ());
  else if (!pdb_path.empty ())
    {
      path = pdb_path;
      if (found (path, "Using PDB path from RSDS"))
	return path;
    }

  if (mainline && info_verbose)
    {
      if (!pdb_path.empty ())
	gdb_printf ("PDB file not found for RSDS: %s\n", pdb_path.c_str ());
      else
	gdb_printf ("PDB file not found for EXE: %s\n", exe_path);
    }

  return "";
}

#if GDB_SELF_TEST

namespace selftests
{

/* A PE image of one section, at RVA 0x1000, holding a debug directory of
   one CodeView entry at offset 0 and the RSDS record of GUID, AGE and
   PATH after it, at offset 0x1c.  The file is 0x600 bytes.  */

struct test_image
{
  test_image (const std::array<gdb_byte, 16> &guid, uint32_t age,
	      const std::string &path)
  {
    image.image_base = 0x140000000;
    image.debug_dir_rva = 0x1000;
    image.debug_dir_size = 28;
    image.file_size = 0x600;
    image.sections.push_back ({ image.image_base + 0x1000, 0x400 });

    gdb::byte_vector &data = contents;
    data.assign (0x400, 0);
    /* IMAGE_DEBUG_TYPE_CODEVIEW, the record's size and RVA.  */
    put_u32 (data, 12, 2);
    put_u32 (data, 16, 24 + path.size () + 1);
    put_u32 (data, 20, 0x1000 + 0x1c);
    memcpy (&data[0x1c], "RSDS", 4);
    memcpy (&data[0x1c + 4], guid.data (), guid.size ());
    put_u32 (data, 0x1c + 20, age);
    memcpy (&data[0x1c + 24], path.c_str (), path.size ());
  }

  /* The RSDS record pdb_parse_rsds finds.  */
  std::optional<pdb_rsds_info> parse () const
  {
    auto read = [&] (size_t i, gdb_byte *buf, uint64_t offset,
		     uint64_t len)
      {
	if (i != 0 || offset > contents.size ()
	    || len > contents.size () - offset)
	  return false;
	memcpy (buf, contents.data () + offset, len);
	return true;
      };
    auto read_file = [&] (uint64_t offset, gdb_byte *buf, uint64_t len)
      {
	if (offset > contents.size () || len > contents.size () - offset)
	  return false;
	memcpy (buf, contents.data () + offset, len);
	return true;
      };
    pdb_pe_image copy = image;
    copy.read = read;
    copy.read_file = read_file;
    return pdb_parse_rsds (copy);
  }

  pdb_pe_image image;
  gdb::byte_vector contents;
};

/* pdb_parse_rsds reads the GUID, age and PDB path of the CodeView entry,
   and gives nothing for a debug directory or CodeView record that lies
   outside the file or its section, or is too small or too large.  */

static void
test_rsds_info ()
{
  std::array<gdb_byte, 16> guid;
  for (int i = 0; i < 16; i++)
    guid[i] = i + 1;
  const std::string path = "C:\\build\\prog.pdb";
  const test_image good (guid, 7, path);

  std::optional<pdb_rsds_info> rsds = good.parse ();
  SELF_CHECK (rsds.has_value ());
  SELF_CHECK (rsds->guid == guid);
  SELF_CHECK (rsds->age == 7);
  SELF_CHECK (rsds->pdb_path == path);

  /* No debug directory, or one of no bytes.  */
  test_image t = good;
  t.image.debug_dir_rva = 0;
  SELF_CHECK (!t.parse ().has_value ());
  t = good;
  t.image.debug_dir_size = 0;
  SELF_CHECK (!t.parse ().has_value ());

  /* A debug directory larger than the file, than its section, or outside
     every section.  */
  for (uint32_t size : { 0xf0000000u, 0x401u })
    {
      t = good;
      t.image.debug_dir_size = size;
      SELF_CHECK (!t.parse ().has_value ());
    }
  t = good;
  t.image.debug_dir_rva = 0x5000;
  SELF_CHECK (!t.parse ().has_value ());

  /* An entry of another type, a CodeView record too small for a path,
     larger than its section or the file, or outside every section, is
     skipped.  */
  auto patched = [&] (size_t offset, uint32_t v)
    {
      test_image p = good;
      put_u32 (p.contents, offset, v);
      return p.parse ();
    };
  SELF_CHECK (!patched (12, 4).has_value ());
  for (uint32_t size : { 24u, 0x500u, 0x1000u, 0x10000u })
    SELF_CHECK (!patched (16, size).has_value ());
  for (uint32_t rva : { 0u, 0x5000u })
    SELF_CHECK (!patched (20, rva).has_value ());

  /* A CodeView record longer than the longest path needs is skipped even
     when its section holds it and the file size is unknown; one just
     within that length is read.  */
  test_image big = good;
  big.image.file_size = 0;
  big.image.sections[0].size = 0x10000;
  big.contents.resize (0x10000, 0);
  put_u32 (big.contents, 16, 0x9000);
  SELF_CHECK (!big.parse ().has_value ());
  put_u32 (big.contents, 16, 0x8000);
  rsds = big.parse ();
  SELF_CHECK (rsds.has_value () && rsds->pdb_path == path);

  /* A record in no section is read at PointerToRawData, within the file
     only.  */
  auto at_file_offset = [&] (uint32_t ptr)
    {
      test_image p = good;
      put_u32 (p.contents, 20, 0);
      put_u32 (p.contents, 24, ptr);
      return p.parse ();
    };
  rsds = at_file_offset (0x1c);
  SELF_CHECK (rsds.has_value () && rsds->age == 7 && rsds->pdb_path == path);
  SELF_CHECK (!at_file_offset (0x5f0).has_value ());
  SELF_CHECK (!at_file_offset (0).has_value ());

  /* The second of two entries is found when the first is skipped.  The
     RSDS record moves past the second entry, a copy of the first; the
     first becomes another type.  */
  t = good;
  memmove (&t.contents[0x1c + 28], &t.contents[0x1c], 0x100);
  memcpy (&t.contents[28], &t.contents[0], 28);
  put_u32 (t.contents, 28 + 20, 0x1000 + 0x1c + 28);
  put_u32 (t.contents, 12, 4);
  t.image.debug_dir_size = 56;
  rsds = t.parse ();
  SELF_CHECK (rsds.has_value () && rsds->age == 7);

  /* A CodeView record of another format (NB10).  */
  SELF_CHECK (!patched (0x1c, 0x3031424e).has_value ());

  /* An empty path, and a path without its terminator in the record.  */
  SELF_CHECK (!test_image (guid, 7, "").parse ().has_value ());
  SELF_CHECK (!patched (16, 24 + path.size ()).has_value ());

  /* A section that cannot be read.  */
  t = good;
  t.contents.resize (16);
  SELF_CHECK (!t.parse ().has_value ());
}

/* Where the PDB is looked for: under the RSDS record's base name next to
   the executable, under the executable's own name there, then at the
   RSDS path itself unless that names a network or device path or is not
   absolute on this host.  The files that exist are the paths in a set.  */

static void
test_find_pdb_file ()
{
  std::set<std::string> files;
  auto present = [&] (const std::string &path)
    {
      return files.count (path) != 0;
    };

  /* An absolute path on this host, as the RSDS record spells it and as
     GDB tries it.  */
#ifdef _WIN32
  const char *abs_rsds = "C:\\build\\out\\linked.pdb";
  const char *abs_path = "C:/build/out/linked.pdb";
#else
  const char *abs_rsds = "/build/out/linked.pdb";
  const char *abs_path = "/build/out/linked.pdb";
#endif

  pdb_rsds_info rsds;
  rsds.pdb_path = abs_rsds;
  auto find = [&] ()
    {
      return pdb_find_pdb_file ("dir/prog.exe", rsds, true, present);
    };

  SELF_CHECK (find ().empty ());
  files.insert (abs_path);
  SELF_CHECK (find () == abs_path);
  files.insert ("dir/prog.pdb");
  SELF_CHECK (find () == "dir/prog.pdb");
  files.insert ("dir/linked.pdb");
  SELF_CHECK (find () == "dir/linked.pdb");

  /* For an executable named without a directory, the base name from the
     RSDS record is looked for in the current directory.  */
  files.insert ("linked.pdb");
  SELF_CHECK (pdb_find_pdb_file ("prog.exe", rsds, true, present)
	      == "linked.pdb");
  files.erase ("linked.pdb");

  /* With no RSDS path, the executable's name; for an executable in the
     current directory, there.  */
  pdb_rsds_info none;
  SELF_CHECK (pdb_find_pdb_file ("dir/prog.exe", none, true, present)
	      == "dir/prog.pdb");
  files.insert ("prog.pdb");
  SELF_CHECK (pdb_find_pdb_file ("prog.exe", none, true, present)
	      == "prog.pdb");

  /* A network or device path, or a relative one, is not opened, but its
     base name is still looked for next to the executable.  */
  files = { "//host/share/linked.pdb", "//?/C:/linked.pdb",
	    "out/linked.pdb" };
  for (const char *not_opened : { "\\\\host\\share\\linked.pdb",
				  "//host/share/linked.pdb",
				  "\\\\?\\C:\\linked.pdb",
				  "out\\linked.pdb" })
    {
      rsds.pdb_path = not_opened;
      SELF_CHECK (find ().empty ());
      files.insert ("dir/linked.pdb");
      SELF_CHECK (find () == "dir/linked.pdb");
      files.erase ("dir/linked.pdb");
    }
}

/* The directory and base name of a path, with either separator.  */

static void
test_path_parts ()
{
  SELF_CHECK (pdb_get_dirname ("C:\\a\\b.pdb") == "C:\\a\\");
  SELF_CHECK (pdb_get_dirname ("/a/b/c.pdb") == "/a/b/");
  SELF_CHECK (pdb_get_dirname ("c.pdb").empty ());
  SELF_CHECK (pdb_get_dirname ("a/") == "a/");

  SELF_CHECK (pdb_get_basename ("C:\\a\\b.pdb") == "b.pdb");
  SELF_CHECK (pdb_get_basename ("/a/b.c.pdb", strip_ext::yes) == "b.c");
  SELF_CHECK (pdb_get_basename ("noext", strip_ext::yes) == "noext");
  SELF_CHECK (pdb_get_basename ("a/").empty ());
}

} /* namespace selftests */

#endif /* GDB_SELF_TEST */

} /* namespace pdb */

INIT_GDB_FILE (pdb_path)
{
#if GDB_SELF_TEST
  selftests::register_test ("pdb-path-parts", pdb::selftests::test_path_parts);
  selftests::register_test ("pdb-rsds-info", pdb::selftests::test_rsds_info);
  selftests::register_test ("pdb-find-pdb-file",
			    pdb::selftests::test_find_pdb_file);
#endif
}
