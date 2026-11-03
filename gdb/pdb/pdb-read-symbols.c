/* PDB symbol reader.

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

/* Read module and global symbols.

   References:
     - microsoft-pdb cvinfo.h: https://github.com/microsoft/microsoft-pdb  */

#include "symtab.h"
#include "gdbtypes.h"
#include "objfiles.h"
#include "complaints.h"
#include "pdb/pdb-internal.h"

#include <string.h>
#include <string>
#include <unordered_map>
#include <chrono>
#include "gdbsupport/selftest.h"

namespace pdb
{
/* S_GDATA32 / S_LDATA32 - Global/local data/variable [cvinfo.h: DATASYM32]
       uint32_t typind      TPI type index
       uint32_t offset      offset within section
       uint16_t segment     section number
       char     name[]  */
inline constexpr auto PDB_SYMBOL_VAR_TYPE_OFFS = 0;
inline constexpr auto PDB_SYMBOL_VAR_SECTION_OFFS_OFFS = 4;
inline constexpr auto PDB_SYMBOL_VAR_SECTION_NUM_OFFS = 8;

/* S_PUB32 - Public symbol (name + address, no type info) [cvinfo.h: PUBSYM32]
       uint32_t flags       public flags
       uint32_t offset      offset within section
       uint16_t segment     section number
       char     name[]  */
inline constexpr auto PDB_SYMBOL_PUB_FLAGS_OFFS = 0;
inline constexpr auto PDB_SYMBOL_PUB_SECT_OFFS_OFFS = 4;
inline constexpr auto PDB_SYMBOL_PUB_SECT_NUM_OFFS = 8;
inline constexpr auto PDB_SYMBOL_PUB_NAME_OFFS = 10;

/* S_PUB32 flag bits (CV_PUBSYMFLAGS in cvinfo.h).  */
inline constexpr uint32_t CV_PUBSYMFLAGS_FUNCTION = 0x02;

/* Decode the symbol record header at REC, where END is one past the
   buffer.  Return std::nullopt if fewer than PDB_RECORD_HDR_SIZE bytes
   are left, if LEN is too small to hold the record kind, or if the full
   record extends past END.  */

static std::optional<pdb_sym_record_hdr>
pdb_parse_sym_record_hdr (const gdb_byte *rec, const gdb_byte *end)
{
  if (end - rec < PDB_RECORD_HDR_SIZE)
    return std::nullopt;

  pdb_sym_record_hdr hdr;
  hdr.len = read_u16 (rec + PDB_RECORD_LEN_OFFS);
  hdr.type = read_u16 (rec + PDB_RECORD_TYPE_OFFS);

  if (hdr.len < 2 || rec + hdr.rec_size () > end)
    return std::nullopt;

  return hdr;
}

/* See pdb-internal.h.  */

const char *
pdb_extract_string (const gdb_byte *p, const gdb_byte *end)
{
  if (p >= end)
    return nullptr;
  if (memchr (p, '\0', end - p) == nullptr)
    return nullptr;
  return (const char *) p;
}

/* Minimum number of record-body bytes (the bytes following the 2-byte
   record type) required to safely read RECTYPE's fixed fields.  For a
   record whose name is its last field (pdb_sym_has_trailing_name), this is
   the name's offset plus one.  Returns 0 for records that carry no fixed
   fields.  */

static size_t
pdb_sym_min_body_size (uint16_t rectype)
{
  switch (rectype)
    {
    case S_PUB32:
      return PDB_SYMBOL_PUB_NAME_OFFS + 1;
    case S_GDATA32:
    case S_LDATA32:
      return PDB_SYMBOL_VAR_NAME_OFFS + 1;
    default:
      return 0;
    }
}

/* Whether RECTYPE's last field is a NUL-terminated name starting at
   pdb_sym_min_body_size (RECTYPE) - 1.  */

static bool
pdb_sym_has_trailing_name (uint16_t rectype)
{
  switch (rectype)
    {
    case S_PUB32:
    case S_GDATA32:
    case S_LDATA32:
      return true;
    default:
      return false;
    }
}

/* Whether the BODY_SIZE-byte body at BODY of a RECTYPE record holds its
   fixed fields and, if it has one, its whole name.  */

static bool
pdb_sym_body_valid (uint16_t rectype, const gdb_byte *body, size_t body_size)
{
  size_t min_body = pdb_sym_min_body_size (rectype);
  if (body_size < min_body)
    return false;

  return (!pdb_sym_has_trailing_name (rectype)
	  || memchr (body + min_body - 1, 0, body_size - (min_body - 1))
	       != nullptr);
}

/* CV_CFL_LANG codes carried in the iLanguage field of S_COMPILE*.  */
inline constexpr uint8_t CV_CFL_C = 0x00;
inline constexpr uint8_t CV_CFL_MASM = 0x03;

/* iLanguage is the low byte of the flags dword, which is the first
   field of S_COMPILE*.  */
inline constexpr auto PDB_SYMBOL_COMPILE_FLAGS_OFFS = 0;

/* Map a CV_CFL_LANG code to a GDB language: C++ for CV_CFL_CXX and the
   codes GDB does not tell apart.  */

static enum language
pdb_cv_lang_to_language (uint8_t cv_lang)
{
  switch (cv_lang)
    {
    case CV_CFL_C:
      return language_c;
    case CV_CFL_MASM:
      return language_asm;
    default:
      return language_cplus;
    }
}

/* See pdb-internal.h.  */

enum language
pdb_module_language (pdb_per_objfile *pdb, pdb_module_info *mod_info)
{
  if (mod_info->language != language_unknown)
    return mod_info->language;

  mod_info->language = language_cplus;
  uint16_t stream = mod_info->stream_number;
  uint32_t syms_end = mod_info->sym_byte_size;
  if (stream == MODI_NO_STREAM || stream >= pdb->streams.size ()
      || syms_end > pdb->streams[stream].size)
    return mod_info->language;

  gdb::byte_vector scratch;
  uint32_t off = PDB_MODULE_SYMBOLS_OFFS;
  while (off + PDB_RECORD_HDR_SIZE <= syms_end)
    {
      const gdb_byte *rec
	= pdb_stream_bytes (pdb, stream, off, PDB_RECORD_HDR_SIZE, scratch);
      size_t rec_size = read_u16 (rec) + sizeof (uint16_t);
      if (rec_size > syms_end - off)
	break;

      rec = pdb_stream_bytes (pdb, stream, off, rec_size, scratch);
      auto hdr = pdb_parse_sym_record_hdr (rec, rec + rec_size);
      if (!hdr)
	break;

      if ((hdr->type == S_COMPILE2 || hdr->type == S_COMPILE3)
	  && rec_size - PDB_RECORD_DATA_OFFS > PDB_SYMBOL_COMPILE_FLAGS_OFFS)
	{
	  uint8_t cv_lang = read_u8 (rec + PDB_RECORD_DATA_OFFS
				     + PDB_SYMBOL_COMPILE_FLAGS_OFFS);
	  mod_info->language = pdb_cv_lang_to_language (cv_lang);
	  break;
	}

      off += rec_size;
    }

  return mod_info->language;
}

/* See pdb-internal.h.  */

void
pdb_read_sym_record_stream (pdb_per_objfile *pdb)
{
  uint16_t idx = pdb->sym_record_stream;
  if (idx == 0 || idx == 0xFFFF)
    return;

  /* Move into the owned buffer; GSI lookups reference these bytes
     directly for the objfile's lifetime.  */
  pdb->sym_record_data = pdb->read_stream (idx);
}

/* Validate and parse a GSI hash region at DATA with DATA_SIZE bytes.
   Returns the parsed header on success, std::nullopt on failure.  The data
   pointer must point to the GSIHashHdr (i.e. past any PublicsStreamHeader).  */

static std::optional<pdb_gsi_hdr>
pdb_parse_gsi_hash_header (gdb_byte *data, uint32_t data_size)
{
  if (data_size < GSI_HASH_HDR_SIZE)
    return std::nullopt;

  pdb_gsi_hdr hdr = {};
  hdr.sig = read_u32 (data + 0);
  hdr.ver = read_u32 (data + 4);
  uint32_t hr_bytes = read_u32 (data + 8);
  uint32_t bucket_bytes = read_u32 (data + 12);

  /* Older layouts lay the hash records and buckets out differently.  */
  if (hdr.sig != GSI_HASH_HDR_SIGNATURE || hdr.ver != GSI_HASH_HDR_VERSION)
    {
      pdb_warning ("GSI hash header has signature 0x%x and version 0x%x,"
		   " which are not read", hdr.sig, hdr.ver);
      return std::nullopt;
    }

  uint64_t needed = (uint64_t) GSI_HASH_HDR_SIZE + hr_bytes + bucket_bytes;
  if (needed > data_size)
    {
      pdb_warning ("GSI hash data truncated: need %" PRIu64 " bytes,"
		   " have %u", needed, data_size);
      return std::nullopt;
    }

  hdr.hr = gdb::make_array_view (data + GSI_HASH_HDR_SIZE, hr_bytes);
  return hdr;
}

/* See pdb-internal.h.  */

std::optional<pdb_sym_record_hdr>
pdb_psgsi_hash_record (const pdb_per_objfile *pdb,
		       gdb::array_view<const gdb_byte> hr, uint32_t i,
		       uint32_t *offset)
{
  const gdb_byte *rec = hr.data () + i * GSI_HASH_RECORD_SIZE;
  uint32_t offs = read_u32 (rec + GSI_HASH_RECORD_SYMOFFS_OFFS);
  if (offs == 0)
    return std::nullopt;

  offs -= 1;
  size_t size = pdb->sym_record_data.size ();
  if (offs > size || size - offs < PDB_RECORD_HDR_SIZE)
    {
      pdb_complaint ("PSGSI hash record %u: offset %u out of range "
		     "(sym_record_size=%zu)", i, offs, size);
      return std::nullopt;
    }

  /* The whole record must lie in the stream.  */
  const gdb_byte *syms = pdb->sym_record_data.data ();
  auto hdr = pdb_parse_sym_record_hdr (syms + offs, syms + size);
  if (!hdr)
    {
      pdb_complaint ("PSGSI hash record %u: bad record header at "
		     "offset %u", i, offs);
      return std::nullopt;
    }

  *offset = offs;
  return hdr;
}

/* See pdb-internal.h.  */

std::optional<pdb_public>
pdb_psgsi_public (const pdb_per_objfile *pdb,
		  gdb::array_view<const gdb_byte> hr, uint32_t i)
{
  uint32_t offs;
  auto hdr = pdb_psgsi_hash_record (pdb, hr, i, &offs);
  if (!hdr || hdr->type != S_PUB32)
    return std::nullopt;

  const gdb_byte *sym_start = pdb->sym_record_data.data () + offs;
  uint16_t reclen = hdr->len;
  size_t rec_size = hdr->rec_size ();

  /* S_PUB32 body must contain at least flags + sect_offs + sect_num
     + one byte for the (possibly empty) NUL-terminated name.  */
  const gdb_byte *rec_data = sym_start + PDB_RECORD_DATA_OFFS;
  const gdb_byte *rec_end = sym_start + rec_size;
  if (rec_data + PDB_SYMBOL_PUB_NAME_OFFS + 1 > rec_end)
    {
      pdb_complaint ("PSGSI hash record %u: S_PUB32 body truncated"
		     " (reclen=%u)", i, reclen);
      return std::nullopt;
    }

  pdb_public pub;
  pub.flags = read_u32 (rec_data + PDB_SYMBOL_PUB_FLAGS_OFFS);
  pub.offset = read_u32 (rec_data + PDB_SYMBOL_PUB_SECT_OFFS_OFFS);
  pub.section = read_u16 (rec_data + PDB_SYMBOL_PUB_SECT_NUM_OFFS);
  pub.name = pdb_extract_string (rec_data + PDB_SYMBOL_PUB_NAME_OFFS,
				 rec_end);
  if (pub.name == nullptr)
    {
      pdb_complaint ("PSGSI hash record %u: S_PUB32 name not"
		     " NUL-terminated", i);
      return std::nullopt;
    }

  /* minimal_symbol_reader strips the target's leading symbol
     character from a name; for x86-64 PE that character is NUL, which
     an empty name's first character matches, and stripping it throws.  */
  if (*pub.name == '\0')
    {
      pdb_complaint ("PSGSI hash record %u: S_PUB32 without a name", i);
      return std::nullopt;
    }

  if (!pdb->section_valid (pub.section))
    return std::nullopt;

  /* A public may sit at the end of its section, as a linker's end
     marker does, but not past it.  */
  if (!pdb->range_in_section (pub.section, pub.offset, 0))
    {
      pdb_complaint ("S_PUB32 %s: offset 0x%x is past the end of"
		     " section %u", pub.name, pub.offset, pub.section);
      return std::nullopt;
    }

  return pub;
}

/* The natural names of the data objects that the S_GDATA32 and S_LDATA32
   records of PDB's symbol record stream describe, keyed by unrelocated
   address.  A record cut short, one whose name has no terminator or is
   empty, and one whose offset is not in its section give no name.  */

static std::unordered_map<CORE_ADDR, const char *>
pdb_data_names (pdb_per_objfile *pdb)
{
  std::unordered_map<CORE_ADDR, const char *> names;
  const gdb_byte *rec = pdb->sym_record_data.data ();
  const gdb_byte *rec_end = rec + pdb->sym_record_data.size ();

  while (rec_end - rec >= PDB_RECORD_HDR_SIZE)
    {
      auto hdr = pdb_parse_sym_record_hdr (rec, rec_end);
      if (!hdr)
	break;

      size_t rec_size = hdr->rec_size ();
      const gdb_byte *body = rec + PDB_RECORD_DATA_OFFS;

      if ((hdr->type == S_GDATA32 || hdr->type == S_LDATA32)
	  && pdb_sym_body_valid (hdr->type, body,
				 rec_size - PDB_RECORD_DATA_OFFS))
	{
	  uint32_t sect_offs
	    = read_u32 (body + PDB_SYMBOL_VAR_SECTION_OFFS_OFFS);
	  uint16_t sect_num = read_u16 (body + PDB_SYMBOL_VAR_SECTION_NUM_OFFS);
	  const char *name
	    = pdb_extract_string (body + PDB_SYMBOL_VAR_NAME_OFFS,
				  rec + rec_size);

	  if (pdb->range_in_section (sect_num, sect_offs, 0)
	      && name != nullptr && *name != '\0')
	    names.emplace
	      (pdb->map_section_offset_unrelocated (sect_num, sect_offs),
	       name);
	}

      rec += rec_size;
    }

  return names;
}

void
pdb_build_minsyms (pdb_per_objfile *pdb)
{
  if (pdb->sym_record_data.empty ())
    return;

  if (pdb->psgsi_stream == 0 || pdb->psgsi_stream == 0xFFFF)
    return;

  auto minsym_start = std::chrono::steady_clock::now ();

  /* PSGSI stream, used only here and freed at scope exit.  */
  auto data_buf = pdb->read_stream (pdb->psgsi_stream);
  if (data_buf.size () < PSGSI_HDR_SIZE)
    return;

  gdb_byte *data = data_buf.data ();

  gdb_byte *gsi_data = data + PSGSI_HDR_SIZE;

  uint32_t sym_hash_size = read_u32 (data + PSGSI_HDR_SYM_HASH_OFFS);
  size_t gsi_avail = data_buf.size () - PSGSI_HDR_SIZE;
  if (sym_hash_size > gsi_avail)
    return;

  auto gsi = pdb_parse_gsi_hash_header (gsi_data, sym_hash_size);
  if (!gsi)
    return;

  minimal_symbol_reader reader (pdb->objfile);

  /* A public carries only the decorated name and there is no MSVC
     demangler, so a data object that has both is better shown by its
     natural name.  */
  std::unordered_map<CORE_ADDR, const char *> data_names
    = pdb_data_names (pdb);

  /* Walk GSI hash records (microsoft-pdb gsi.h: HRFile).
     Each record is GSI_HASH_RECORD_SIZE (8) bytes:
       uint32_t offs: SymRecordStream byte offset, stored as offset+1
		      so that 0 can mean "empty slot"
       uint32_t cref: reference count (unused here)  */
  for (uint32_t i = 0; i < gsi->hr.size () / GSI_HASH_RECORD_SIZE; i++)
    {
      std::optional<pdb_public> pub = pdb_psgsi_public (pdb, gsi->hr, i);
      if (!pub)
	continue;
      const char *name = pub->name;
      uint32_t pub_flags = pub->flags;
      uint16_t sect_num = pub->section;
      uint32_t sect_offs = pub->offset;

      /* GDB adds the section offsets to a minimal symbol itself, and a DLL's
	 are already set when its symbols are read.  */
      CORE_ADDR unrel = pdb->map_section_offset_unrelocated (sect_num,
							     sect_offs);

      /* "??_C@" names an MSVC string literal, which ELF keeps as a local
	 .LC label outside the symbol table; as a minimal symbol it would
	 label every pointer to a literal ("<??_C@...> \"text\"").  */
      if (startswith (name, "??_C@"))
	continue;

      const bool decorated = name[0] == '?';

      /* CV_PUBSYMFLAGS cvpsfFunction: the public is code.  */
      const bool is_function = (pub_flags & CV_PUBSYMFLAGS_FUNCTION) != 0;
      enum minimal_symbol_type type = is_function ? mst_text : mst_data;
      const char *msym_name = name;

      if (!is_function)
	{
	  auto nat = data_names.find (unrel);
	  if (nat != data_names.end () && decorated)
	    msym_name = nat->second;
	}

      int sect_idx = sect_num - 1;
      reader.record_with_info (msym_name, unrelocated_addr (unrel), type,
			       sect_idx);
    }

  double minsym_ms
    = std::chrono::duration<double, std::milli>
	(std::chrono::steady_clock::now () - minsym_start).count ();
  pdb_dbg_printf ("Recorded %zu minimal symbols in %.2f ms",
		  reader.count (), minsym_ms);
  reader.install ();

  /* As coffread does for a PE symbol table: the function FOO that has an
     __imp_FOO import address slot is the import thunk.  */
  for (minimal_symbol *msym : pdb->objfile->msymbols ())
    {
      const char *name = msym->linkage_name ();
      if (msym->type () != mst_data || !startswith (name, "__imp_"))
	continue;
      bound_minimal_symbol thunk
	= lookup_minimal_symbol (current_program_space, name + 6,
				 pdb->objfile);
      if (thunk.minsym != nullptr && thunk.minsym->type () == mst_text)
	thunk.minsym->set_type (mst_solib_trampoline);
    }
}

#if GDB_SELF_TEST

namespace selftests {

/* A symbol record's header: its length counts the bytes after it, at
   least the kind, and the record must end inside the buffer.  */

static void
test_sym_record_hdr ()
{
  gdb::byte_vector rec { 0x06, 0x00, 0x0d, 0x11, 1, 2, 3, 4 };
  const gdb_byte *end = rec.data () + rec.size ();
  std::optional<pdb_sym_record_hdr> hdr
    = pdb_parse_sym_record_hdr (rec.data (), end);
  SELF_CHECK (hdr.has_value ());
  SELF_CHECK (hdr->len == 6 && hdr->type == 0x110d && hdr->rec_size () == 8);

  /* Ending one byte past the buffer.  */
  SELF_CHECK (!pdb_parse_sym_record_hdr (rec.data (), end - 1).has_value ());
  /* A length too small for the kind.  */
  for (gdb_byte len : { 0, 1 })
    {
      gdb::byte_vector small = rec;
      small[0] = len;
      SELF_CHECK (!pdb_parse_sym_record_hdr (small.data (),
					     small.data () + small.size ())
		  .has_value ());
    }
  /* Fewer than 4 bytes left.  */
  for (size_t left = 0; left < 4; left++)
    SELF_CHECK (!pdb_parse_sym_record_hdr (rec.data (), rec.data () + left)
		.has_value ());
  /* A length of 0xffff past a short buffer.  */
  rec[0] = 0xff;
  rec[1] = 0xff;
  SELF_CHECK (!pdb_parse_sym_record_hdr (rec.data (), end).has_value ());
}

/* A PSGSI hash record names a record of the symbol record stream by its
   offset plus one: the offset is file data, so the record must lie in the
   stream, also when the offset is so large that adding the header size
   wraps past 2^32.  */

static void
test_psgsi_hash_record ()
{
  pdb_per_objfile pdb (nullptr);
  /* Two records of 8 bytes: an S_PUB32 cut short is fine here, only the
     header is read.  */
  pdb.sym_record_data = { 0x06, 0x00, 0x0e, 0x11, 1, 2, 3, 4,
			  0x06, 0x00, 0x0e, 0x11, 5, 6, 7, 0 };
  const uint32_t size = pdb.sym_record_data.size ();

  gdb::byte_vector hr;
  for (uint32_t stored : { 0u, 1u, 9u, 0xfffffffeu, 0xffffffffu, size + 1,
			   size - 2, 6u })
    {
      put_u32 (hr, hr.size (), stored);
      put_u32 (hr, hr.size (), 1);
    }

  uint32_t offset = 0;
  /* An empty slot.  */
  SELF_CHECK (!pdb_psgsi_hash_record (&pdb, hr, 0, &offset).has_value ());

  /* The two records.  */
  for (uint32_t i : { 1u, 2u })
    {
      auto hdr = pdb_psgsi_hash_record (&pdb, hr, i, &offset);
      SELF_CHECK (hdr.has_value () && hdr->type == 0x110e
		  && hdr->rec_size () == 8 && offset == (i - 1) * 8);
    }

  /* Offsets past the stream, at its end, and with fewer bytes left than
     a header takes.  */
  for (uint32_t i : { 3u, 4u, 5u, 6u })
    SELF_CHECK (!pdb_psgsi_hash_record (&pdb, hr, i, &offset).has_value ());

  /* A header whose record runs past the stream.  */
  SELF_CHECK (!pdb_psgsi_hash_record (&pdb, hr, 7, &offset).has_value ());
}

/* Append to PDB's symbol record stream a record of KIND whose body is
   BODY, and to HR the hash record naming it.  */

static void
add_sym_record (pdb_per_objfile &pdb, gdb::byte_vector &hr, uint16_t kind,
		const gdb::byte_vector &body)
{
  gdb::byte_vector &syms = pdb.sym_record_data;
  uint32_t offset = syms.size ();
  uint16_t len = body.size () + 2;
  syms.push_back (len & 0xff);
  syms.push_back (len >> 8);
  syms.push_back (kind & 0xff);
  syms.push_back (kind >> 8);
  syms.insert (syms.end (), body.begin (), body.end ());
  put_u32 (hr, hr.size (), offset + 1);
  put_u32 (hr, hr.size (), 1);
}

/* The body of an S_PUB32 of FLAGS at SECTION:OFFSET named NAME, its
   terminator left out unless TERMINATED.  */

static gdb::byte_vector
pub32_body (uint32_t flags, uint16_t section, uint32_t offset,
	    const std::string &name, bool terminated = true)
{
  gdb::byte_vector body;
  put_u32 (body, PDB_SYMBOL_PUB_FLAGS_OFFS, flags);
  put_u32 (body, PDB_SYMBOL_PUB_SECT_OFFS_OFFS, offset);
  body.push_back (section & 0xff);
  body.push_back (section >> 8);
  body.insert (body.end (), name.begin (), name.end ());
  if (terminated)
    body.push_back (0);
  return body;
}

/* The publics of the PSGSI: an S_PUB32 is read only when it is whole, its
   name has its terminator and is not empty, and its section is one of the
   image's.  */

static void
test_psgsi_public ()
{
  pdb_per_objfile pdb (nullptr);
  pdb.section_addresses = { 0x140001000, 0x140002000 };
  pdb.section_sizes = { 0x100, 0x200 };

  gdb::byte_vector hr;
  add_sym_record (pdb, hr, S_PUB32, pub32_body (2, 1, 0x10, "alive"));
  add_sym_record (pdb, hr, S_PUB32, pub32_body (0, 1, 0x10, ""));
  add_sym_record (pdb, hr, S_PUB32, pub32_body (0, 1, 0x10, "open", false));
  gdb::byte_vector cut = pub32_body (0, 1, 0x10, "");
  cut.resize (PDB_SYMBOL_PUB_NAME_OFFS);
  add_sym_record (pdb, hr, S_PUB32, cut);
  add_sym_record (pdb, hr, S_PUB32, pub32_body (0, 3, 0x10, "nosect"));
  add_sym_record (pdb, hr, S_PUB32, pub32_body (0, 0, 0x10, "zero"));
  add_sym_record (pdb, hr, S_GDATA32, pub32_body (0, 1, 0x10, "data"));
  put_u32 (hr, hr.size (), 0);
  put_u32 (hr, hr.size (), 0);

  std::optional<pdb_public> pub = pdb_psgsi_public (&pdb, hr, 0);
  SELF_CHECK (pub.has_value () && strcmp (pub->name, "alive") == 0
	      && pub->flags == 2 && pub->section == 1 && pub->offset == 0x10);

  /* An empty name, a name without its terminator, a record cut short, a
     section the image does not have, another kind and an empty slot.  */
  for (uint32_t i = 1; i < 8; i++)
    SELF_CHECK (!pdb_psgsi_public (&pdb, hr, i).has_value ());
}

/* A public may sit at the end of its section, as a linker's end marker
   does, but not past it.  */

static void
test_psgsi_public_range ()
{
  pdb_per_objfile pdb (nullptr);
  pdb.section_addresses = { 0x140001000, 0x140002000 };
  pdb.section_sizes = { 0x100, 0x200 };

  gdb::byte_vector hr;
  add_sym_record (pdb, hr, S_PUB32, pub32_body (0, 2, 0x200, "end"));
  add_sym_record (pdb, hr, S_PUB32, pub32_body (0, 1, 0x101, "far"));
  add_sym_record (pdb, hr, S_PUB32, pub32_body (0, 1, 0xffffffff, "wrap"));

  SELF_CHECK (pdb_psgsi_public (&pdb, hr, 0).has_value ());
  SELF_CHECK (!pdb_psgsi_public (&pdb, hr, 1).has_value ());
  SELF_CHECK (!pdb_psgsi_public (&pdb, hr, 2).has_value ());
}

/* A GSI hash header is read only in the layout of GSIHashSCImpvV70, and
   only when the hash records and buckets it counts are in the data.  */

static void
test_gsi_hash_header ()
{
  auto header = [] (uint32_t sig, uint32_t ver, uint32_t hr_bytes,
		    uint32_t bucket_bytes)
    {
      gdb::byte_vector data;
      put_u32 (data, 0, sig);
      put_u32 (data, 4, ver);
      put_u32 (data, 8, hr_bytes);
      put_u32 (data, 12, bucket_bytes);
      data.resize (GSI_HASH_HDR_SIZE + 16, 0);
      return data;
    };

  gdb::byte_vector data = header (GSI_HASH_HDR_SIGNATURE,
				  GSI_HASH_HDR_VERSION, 8, 8);
  auto hdr = pdb_parse_gsi_hash_header (data.data (), data.size ());
  SELF_CHECK (hdr.has_value () && hdr->hr.size () == 8
	      && hdr->hr.data () == data.data () + GSI_HASH_HDR_SIZE);

  /* Another signature, another version, more records or buckets than
     the data holds, and data shorter than the header.  */
  data = header (0, GSI_HASH_HDR_VERSION, 8, 8);
  SELF_CHECK (!pdb_parse_gsi_hash_header (data.data (), data.size ()));
  data = header (GSI_HASH_HDR_SIGNATURE, 0x12345678, 8, 8);
  SELF_CHECK (!pdb_parse_gsi_hash_header (data.data (), data.size ()));
  data = header (GSI_HASH_HDR_SIGNATURE, GSI_HASH_HDR_VERSION, 8, 9);
  SELF_CHECK (!pdb_parse_gsi_hash_header (data.data (), data.size ()));
  data = header (GSI_HASH_HDR_SIGNATURE, GSI_HASH_HDR_VERSION,
		 0xffffffff, 0xffffffff);
  SELF_CHECK (!pdb_parse_gsi_hash_header (data.data (), data.size ()));
  SELF_CHECK (!pdb_parse_gsi_hash_header (data.data (),
					  GSI_HASH_HDR_SIZE - 1));
}

/* Every record kind the reader reads has a minimum body, and a kind whose
   last field is a name needs the name's terminator inside the record.  */

static void
test_sym_body_valid ()
{
  const uint16_t kinds[] = { S_PUB32, S_GDATA32, S_LDATA32 };

  for (uint16_t kind : kinds)
    {
      size_t min = pdb_sym_min_body_size (kind);
      SELF_CHECK (min > 0);

      /* One byte short.  */
      gdb::byte_vector body (min - 1, 0);
      SELF_CHECK (!pdb_sym_body_valid (kind, body.data (), body.size ()));

      /* The minimum: an empty name, or the fixed fields alone.  */
      body.assign (min, 0);
      SELF_CHECK (pdb_sym_body_valid (kind, body.data (), body.size ()));

      if (!pdb_sym_has_trailing_name (kind))
	continue;

      /* A name running to the end of the record.  */
      body.assign (min + 8, 'A');
      SELF_CHECK (!pdb_sym_body_valid (kind, body.data (), body.size ()));
      body.back () = 0;
      SELF_CHECK (pdb_sym_body_valid (kind, body.data (), body.size ()));
    }

  /* A kind the reader does not read needs no body.  */
  SELF_CHECK (pdb_sym_body_valid (0xffff, nullptr, 0));
}

/* A string is read only when its terminator is inside the buffer.  */

static void
test_extract_string ()
{
  const gdb_byte name[] = { 'a', 'b', 'c', 0, 'd' };
  SELF_CHECK (pdb_extract_string (name, name + 4) == (const char *) name);
  SELF_CHECK (pdb_extract_string (name, name + 3) == nullptr);
  SELF_CHECK (pdb_extract_string (name + 3, name + 4)
	      == (const char *) name + 3);
  SELF_CHECK (pdb_extract_string (name + 4, name + 5) == nullptr);
  SELF_CHECK (pdb_extract_string (name, name) == nullptr);
  SELF_CHECK (pdb_extract_string (name + 4, name) == nullptr);
}

/* The body of an S_GDATA32 or S_LDATA32 at SECTION:OFFSET named NAME, its
   terminator left out unless TERMINATED.  */

static gdb::byte_vector
data32_body (uint16_t section, uint32_t offset, const std::string &name,
	     bool terminated = true)
{
  gdb::byte_vector body;
  put_u32 (body, PDB_SYMBOL_VAR_TYPE_OFFS, 0x74);
  put_u32 (body, PDB_SYMBOL_VAR_SECTION_OFFS_OFFS, offset);
  body.push_back (section & 0xff);
  body.push_back (section >> 8);
  body.insert (body.end (), name.begin (), name.end ());
  if (terminated)
    body.push_back (0);
  return body;
}

/* The data records name the objects in their sections: a record past the
   end of its section, in no section of the image, cut short, or with an
   empty or unterminated name names nothing, and the first record at an
   address gives its name.  */

static void
test_data_names ()
{
  pdb_per_objfile pdb (nullptr);
  pdb.section_addresses = { 0x140001000, 0x140002000 };
  pdb.section_sizes = { 0x100, 0x200 };

  gdb::byte_vector hr;
  add_sym_record (pdb, hr, S_GDATA32, data32_body (1, 0x10, "global"));
  add_sym_record (pdb, hr, S_LDATA32, data32_body (2, 0x200, "at_end"));
  add_sym_record (pdb, hr, S_GDATA32, data32_body (2, 0x201, "past_end"));
  add_sym_record (pdb, hr, S_GDATA32, data32_body (1, 0xffffffff, "wrap"));
  add_sym_record (pdb, hr, S_GDATA32, data32_body (3, 0x10, "nosect"));
  add_sym_record (pdb, hr, S_GDATA32, data32_body (0, 0x10, "zero"));
  add_sym_record (pdb, hr, S_GDATA32, data32_body (1, 0x20, ""));
  add_sym_record (pdb, hr, S_GDATA32, data32_body (1, 0x30, "open", false));
  add_sym_record (pdb, hr, S_LDATA32, data32_body (1, 0x10, "second"));
  add_sym_record (pdb, hr, S_PUB32, pub32_body (0, 1, 0x40, "public"));
  gdb::byte_vector cut = data32_body (1, 0x50, "");
  cut.resize (PDB_SYMBOL_VAR_NAME_OFFS);
  add_sym_record (pdb, hr, S_GDATA32, cut);

  auto names = pdb_data_names (&pdb);
  SELF_CHECK (names.size () == 2);

  auto global = names.find (0x140001010);
  SELF_CHECK (global != names.end ()
	      && strcmp (global->second, "global") == 0);
  auto at_end = names.find (0x140002200);
  SELF_CHECK (at_end != names.end ()
	      && strcmp (at_end->second, "at_end") == 0);
}

/* The minimal symbols of a PSGSI's publics: a global variable's under its
   source name, from its global data record; none for a string literal or
   a public without a name; a function that has an __imp_ import address
   slot is an import thunk.  */

static void
test_minsyms ()
{
  /* The symbol records, collected without an objfile.  */
  pdb_per_objfile pdb (nullptr);
  gdb::byte_vector hr;
  add_sym_record (pdb, hr, S_PUB32,
		  pub32_body (0, 1, 0x100, "?g_counter@@3HA"));
  add_sym_record (pdb, hr, S_GDATA32, data32_body (1, 0x100, "g_counter"));
  add_sym_record (pdb, hr, S_PUB32,
		  pub32_body (0, 1, 0x200, "??_C@_05ABCDEF@hello@"));
  add_sym_record (pdb, hr, S_PUB32,
		  pub32_body (CV_PUBSYMFLAGS_FUNCTION, 1, 0x300, "Import"));
  add_sym_record (pdb, hr, S_PUB32, pub32_body (0, 1, 0x400, "__imp_Import"));
  add_sym_record (pdb, hr, S_PUB32, pub32_body (0, 1, 0x500, ""));

  /* The PSGSI: its header, the GSI hash header and the hash records, with
     no buckets.  */
  gdb::byte_vector gsi;
  put_u32 (gsi, 0, GSI_HASH_HDR_SIGNATURE);
  put_u32 (gsi, 4, GSI_HASH_HDR_VERSION);
  put_u32 (gsi, 8, hr.size ());
  put_u32 (gsi, 12, 0);
  gsi.insert (gsi.end (), hr.begin (), hr.end ());
  gdb::byte_vector psgsi (PSGSI_HDR_SIZE, 0);
  put_u32 (psgsi, PSGSI_HDR_SYM_HASH_OFFS, gsi.size ());
  psgsi.insert (psgsi.end (), gsi.begin (), gsi.end ());

  test_pdb with_psgsi;
  with_psgsi.streams.resize (5);
  with_psgsi.streams.push_back (psgsi);
  with_psgsi.build ();
  test_module_reader r2 (with_psgsi, 1);
  if (r2.pdb == nullptr)
    return;
  r2.pdb->sym_record_data = pdb.sym_record_data;
  r2.pdb->psgsi_stream = 5;
  pdb_build_minsyms (r2.pdb);

  objfile *objf = r2.obj.objfile;
  bound_minimal_symbol var
    = lookup_minimal_symbol (current_program_space, "g_counter", objf);
  SELF_CHECK (var.minsym != nullptr);
  SELF_CHECK (lookup_minimal_symbol (current_program_space, "?g_counter@@3HA",
				     objf).minsym == nullptr);
  bound_minimal_symbol thunk
    = lookup_minimal_symbol (current_program_space, "Import", objf);
  SELF_CHECK (thunk.minsym != nullptr
	      && thunk.minsym->type () == mst_solib_trampoline);
  for (minimal_symbol *msym : objf->msymbols ())
    SELF_CHECK (!startswith (msym->linkage_name (), "??_C@")
		&& msym->linkage_name ()[0] != '\0');
}

} /* namespace selftests */

#endif /* GDB_SELF_TEST */

} /* namespace pdb */

INIT_GDB_FILE (pdb_read_symbols)
{
#if GDB_SELF_TEST
  selftests::register_test ("pdb-sym-record-hdr",
			    pdb::selftests::test_sym_record_hdr);
  selftests::register_test ("pdb-psgsi-hash-record",
			    pdb::selftests::test_psgsi_hash_record);
  selftests::register_test ("pdb-psgsi-public",
			    pdb::selftests::test_psgsi_public);
  selftests::register_test ("pdb-psgsi-public-range",
			    pdb::selftests::test_psgsi_public_range);
  selftests::register_test ("pdb-gsi-hash-header",
			    pdb::selftests::test_gsi_hash_header);
  selftests::register_test ("pdb-sym-body-valid",
			    pdb::selftests::test_sym_body_valid);
  selftests::register_test ("pdb-extract-string",
			    pdb::selftests::test_extract_string);
  selftests::register_test ("pdb-data-names",
			    pdb::selftests::test_data_names);
  selftests::register_test ("pdb-minsyms",
			    pdb::selftests::test_minsyms);
#endif
}
