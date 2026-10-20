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
#include "buildsym.h"
#include "complaints.h"
#include "block.h"
#include "pdb/pdb-internal.h"
#include "event-top.h"
#include "gdbsupport/scoped_restore.h"

#include <string.h>
#include <string>
#include <unordered_map>
#include <chrono>

namespace pdb
{

/* Parser for a CodeView symbol stream.  */

class pdb_sym_parser
{
public:
  pdb_sym_parser (pdb_per_objfile *pdb_, buildsym_compunit *cu_,
		  pdb_range_pair_vec *func_ranges_,
		  enum language lang_,
		  pdb_symbol_lines *sym_lines_)
    : pdb (pdb_),
      cu (cu_),
      func_ranges (func_ranges_),
      m_lang (lang_),
      m_sym_lines (sym_lines_)
  {
  }

  DISABLE_COPY_AND_ASSIGN (pdb_sym_parser);

  enum language lang () const
  {
    return m_lang;
  }

  /* Give the function FUNC, whose code starts at ENTRY_PC, the symtab of
     its source file once the compunit is built; see pdb_symbol_lines.  */
  void queue_function_file (symbol *func, CORE_ADDR entry_pc)
  {
    if (m_sym_lines == nullptr)
      return;
    auto it = m_sym_lines->at_pc.find (entry_pc);
    if (it == m_sym_lines->at_pc.end ())
      return;
    m_sym_lines->to_apply.push_back ({ func, it->second });
  }

  /* CodeView record handlers.  */
  void handle_func_sym (gdb_byte *rec_data, size_t body_sz, uint16_t rectype);
  void handle_thunk_sym ();
  void handle_scope_end_sym ();

  /* Expose open scopes for end-of-stream recovery.  */

  pdb_scope_stack &scope_stack ()
  {
    return m_scope_stack;
  }

private:

  pdb_per_objfile *pdb;
  buildsym_compunit *cu;
  pdb_range_pair_vec *func_ranges;
  enum language m_lang;

  /* Function declaration lines from the C13 line table, or nullptr.  */
  pdb_symbol_lines *m_sym_lines = nullptr;

  pdb_scope_stack m_scope_stack;
};

/* There are about 200 different symbol types in microsoft-pdb cvinfo.h.
   Below are the supported record types.  */

/* S_GPROC32 / S_LPROC32 - Global/local functions [cvinfo.h: PROCSYM32]
       uint32_t pParent
       uint32_t pEnd
       uint32_t pNext
       uint32_t len          procedure code size
       uint32_t DbgStart
       uint32_t DbgEnd
       uint32_t typind       type index
       uint32_t off          offset in section
       uint16_t seg          section number
       uint8_t  flags
       char     name[]  */
inline constexpr auto PDB_SYMBOL_FUNC_TYPE_OFFS = 24;
inline constexpr auto PDB_SYMBOL_FUNC_CODE_SIZE_OFFS = 12;
inline constexpr auto PDB_SYMBOL_FUNC_SECTION_OFFS_OFFS = 28;
inline constexpr auto PDB_SYMBOL_FUNC_SECTION_NUM_OFFS = 32;
inline constexpr auto PDB_SYMBOL_FUNC_FLAGS_OFFS = 34;
/* Fixed fields plus room for an empty name.  */
inline constexpr auto PDB_SYMBOL_FUNC_MIN_SIZE = PDB_SYMBOL_FUNC_NAME_OFFS + 1;

/* CV_PROCFLAGS bit for a procedure that does not return.  */
inline constexpr uint8_t CV_PFLAG_NEVER = 0x08;
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
inline constexpr auto PDB_SYMBOL_THUNK_NAME_OFFS = 21;

/* Decode a symbol header at REC.  The caller must provide four readable
   header bytes; this function reads them before validating LEN.  END is
   one past the buffer.  Return std::nullopt if LEN is too small or the full
   record extends past END.  */

static std::optional<pdb_sym_record_hdr>
pdb_parse_sym_record_hdr (const gdb_byte *rec, const gdb_byte *end)
{
  pdb_sym_record_hdr hdr;
  hdr.len = read_u16 (rec + PDB_RECORD_LEN_OFFS);
  hdr.type = read_u16 (rec + PDB_RECORD_TYPE_OFFS);

  if (hdr.len < 2 || rec + hdr.rec_size () > end)
    return std::nullopt;

  return hdr;
}

/* Parsed symbol fields shared by the record-specific builders.  */
struct pdb_sym
{
  uint16_t rectype;
  const char *name = nullptr;
  uint32_t type_index = 0;
  pdb_per_objfile *pdb;
  gdb_byte *rec_data;
  buildsym_compunit *cu;
  symbol *sym = nullptr;

protected:
  pdb_sym (gdb_byte *rec_data_in, uint16_t rectype_in, pdb_per_objfile *pdb_in,
	   buildsym_compunit *cu_in = nullptr)
    : rectype (rectype_in),
      pdb (pdb_in),
      rec_data (rec_data_in),
      cu (cu_in)
  {
  }

  /* Set name from a byte offset into rec_data.  */
  void set_name (uint32_t offset)
  {
    name = pdb_canonical_name (pdb, CSTR (rec_data + offset));
  }

  /* Set type_index from a byte offset into rec_data.  */
  void set_type (uint32_t offset)
  {
    type_index = read_u32 (rec_data + offset);
  }

  /* pub_mangled_names is keyed unrelocated, so undo any load bias in ADDR.  */
  CORE_ADDR unrelocated_pub_key (CORE_ADDR addr) const
  {
    return addr - pdb->objfile->text_section_offset ();
  }

  /* Name the symbol.  GDB keeps both a symbol's natural name and its
     linkage name; the CodeView record gives only the natural name,
     NATURAL, and the linkage name is that of an S_PUB32 public symbol at
     ADDR.  The public found there is taken when it is decorated ("?...")
     or is the record's own name, as for a C or extern "C" function;
     otherwise the symbol has NATURAL only.  */

  void set_symbol_names (CORE_ADDR addr, const char *natural)
  {
    natural = pdb_canonical_name (pdb, natural);

    auto it = pdb->pub_mangled_names.find (unrelocated_pub_key (addr));
    if (it != pdb->pub_mangled_names.end ()
	&& (it->second[0] == '?'
	    || (name != nullptr && strcmp (it->second, name) == 0)))
      {
	sym->compute_and_set_names (it->second, true, pdb->objfile->per_bfd);
	/* Set the demangled name from the CodeView record.  */
	const char *demangled
	  = obstack_strdup (&pdb->objfile->objfile_obstack, natural);
	sym->set_demangled_name (demangled, &pdb->objfile->objfile_obstack);
      }
    else
      sym->compute_and_set_names (natural, true, pdb->objfile->per_bfd);
  }

  /* Allocate a symbol using the public name at ADDR when available.  */

  void allocate_symbol_named (domain_enum domain, CORE_ADDR addr,
			      const char *natural)
  {
    sym = pdb->objfile->new_symbol<symbol> ();
    gdb_assert (pdb->cur_parser != nullptr);
    sym->set_language (pdb->cur_parser->lang (),
		       &pdb->objfile->objfile_obstack);
    set_symbol_names (addr, natural);
    sym->set_domain (domain);

    gdb_assert (pdb->cur_parser != nullptr);
  }

public:
};

/* S_GPROC32 / S_LPROC32 / S_GPROC32_ID / S_LPROC32_ID.  */
struct pdb_func_sym final : pdb_sym
{
  uint32_t code_sz;
  uint32_t sect_offs;
  uint16_t sect_num;

  pdb_func_sym (gdb_byte *rec_data, uint16_t rectype, pdb_per_objfile *pdb,
		buildsym_compunit *cu)
    : pdb_sym (rec_data, rectype, pdb, cu)
  {
    set_type (PDB_SYMBOL_FUNC_TYPE_OFFS);
    set_name (PDB_SYMBOL_FUNC_NAME_OFFS);

    /* The type of an _ID record is an IPI id, not a TPI index.  */
    if (rectype == S_GPROC32_ID || rectype == S_LPROC32_ID)
      type_index = 0;

    code_sz = read_u32 (rec_data + PDB_SYMBOL_FUNC_CODE_SIZE_OFFS);
    sect_offs = read_u32 (rec_data + PDB_SYMBOL_FUNC_SECTION_OFFS_OFFS);
    sect_num = read_u16 (rec_data + PDB_SYMBOL_FUNC_SECTION_NUM_OFFS);
    m_flags = read_u8 (rec_data + PDB_SYMBOL_FUNC_FLAGS_OFFS);
  }

  symbol *create_gdb_sym ()
  {
    const auto *of = pdb->objfile;

    /* PDB stores a C++ function name without its parameter list
       (e.g. "ns_a::ns_a_func"), and GDB expects the natural name of a C++
       function to carry one so overloads stay distinct.  C has no
       overloads and its natural name is the bare identifier, so a list
       there would corrupt every frame line and symbol print.  */
    gdb_assert (pdb->cur_parser != nullptr);
    enum language lang = pdb->cur_parser->lang ();

    type *func_type = &pdb_tpi_resolve_type (pdb, type_index);
    const char *natural = pdb_function_name (pdb, name, func_type, lang);

    /* Mark a function that never returns (CV_PFLAG_NEVER, as for exit
       or a [[noreturn]] function), so that "finish" and "return" warn
       that it does not return normally.  GDB keeps the mark in the
       function's type, and functions with the same signature share one
       type, so this function gets its own copy.  */
    if ((m_flags & CV_PFLAG_NEVER) != 0
	&& TYPE_SPECIFIC_FIELD (func_type) == TYPE_SPECIFIC_FUNC)
      {
	type *copy = copy_type (func_type);
	INIT_FUNC_SPECIFIC (copy);
	*copy->main_type->type_specific.func_stuff
	  = *func_type->main_type->type_specific.func_stuff;
	TYPE_NO_RETURN (copy) = 1;
	func_type = copy;
      }

    /* The address also gives the function's linkage name, from a public
       there; see set_symbol_names.  */
    CORE_ADDR addr = pdb->map_section_offset_to_pc (sect_num, sect_offs);
    allocate_symbol_named (FUNCTION_DOMAIN, addr, natural);

    pdb->cur_parser->queue_function_file (sym, addr);
    sym->set_loc_class_index (LOC_BLOCK);
    sym->set_type (func_type);
    sym->set_section_index (SECT_OFF_TEXT (of));
    bool is_global = (rectype == S_GPROC32 || rectype == S_GPROC32_ID);
    auto &list = is_global ? cu->get_global_symbols ()
			   : cu->get_file_symbols ();
    add_symbol_to_list (sym, list);
    return sym;
  }

private:
  uint8_t m_flags;
};

/* Handle S_GPROC32 / S_LPROC32 / S_GPROC32_ID / S_LPROC32_ID.
   Opens a scope; the matching S_END will close it.  */

void
pdb_sym_parser::handle_func_sym (gdb_byte *rec_data, size_t body_size,
				 uint16_t rectype)
{
  if (body_size < PDB_SYMBOL_FUNC_MIN_SIZE)
    {
      pdb_warning ("truncated S_*PROC32 record (body=%zu, need >=%u), "
		   "skipping",
		   body_size, (unsigned) PDB_SYMBOL_FUNC_MIN_SIZE);
      return;
    }

  pdb_func_sym fsym (rec_data, rectype, pdb, cu);

  if (!pdb->section_valid (fsym.sect_num))
    {
      pdb_complaint ("PDB: bad section %u for function '%s'", fsym.sect_num,
		     fsym.name);
      return;
    }

  auto *sym = fsym.create_gdb_sym ();

  CORE_ADDR start = pdb->map_section_offset_to_pc (fsym.sect_num,
						   fsym.sect_offs);
  CORE_ADDR end = start + fsym.code_sz;
  cu->push_context (start);
  if (sym != nullptr)
    cu->set_current_context_function (sym);

  if (func_ranges != nullptr)
    func_ranges->push_back ({ start, end });

  pdb_scope_frame frame { end };
  m_scope_stack.push_back (std::move (frame));
}

/* Handle S_THUNK32.  Opens a scope; the matching S_END will close it.  */

void
pdb_sym_parser::handle_thunk_sym ()
{
  cu->push_context (0);
  m_scope_stack.push_back ({ 0 });
}

/* Handle S_END / S_PROC_ID_END.
   Close the innermost open scope.  */

void
pdb_sym_parser::handle_scope_end_sym ()
{
  if (!m_scope_stack.empty ())
    {
      pdb_scope_frame frame = std::move (m_scope_stack.back ());
      m_scope_stack.pop_back ();
      cu->pop_context (frame.end, nullptr, frame.end != 0);
    }
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
    case S_GPROC32:
    case S_LPROC32:
    case S_GPROC32_ID:
    case S_LPROC32_ID:
      return PDB_SYMBOL_FUNC_NAME_OFFS + 1;
    case S_PUB32:
      return PDB_SYMBOL_PUB_NAME_OFFS + 1;
    case S_GDATA32:
    case S_LDATA32:
      return PDB_SYMBOL_VAR_NAME_OFFS + 1;
    case S_THUNK32:
      return PDB_SYMBOL_THUNK_NAME_OFFS + 1;
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
    case S_GPROC32:
    case S_LPROC32:
    case S_GPROC32_ID:
    case S_LPROC32_ID:
    case S_PUB32:
    case S_GDATA32:
    case S_LDATA32:
    case S_THUNK32:
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
inline constexpr uint8_t CV_CFL_CXX = 0x01;

/* iLanguage is the low byte of the flags dword, which is the first
   field of S_COMPILE*.  */
inline constexpr auto PDB_SYMBOL_COMPILE_FLAGS_OFFS = 0;

/* Map a CV_CFL_LANG code to a GDB language.  */
static enum language
pdb_cv_lang_to_language (uint8_t cv_lang)
{
  switch (cv_lang)
    {
    case CV_CFL_C:
      return language_c;
    case CV_CFL_CXX:
      return language_cplus;
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
  if (stream >= pdb->streams.size ()
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
pdb_parse_symbols (pdb_per_objfile *pdb, pdb_module_info *mod_info,
		   gdb_byte *module_stream, buildsym_compunit *cu,
		   pdb_range_pair_vec *func_ranges,
		   enum language lang, pdb_symbol_lines *sym_lines)
{
  gdb_byte *syms_start = module_stream + PDB_MODULE_SYMBOLS_OFFS;
  const gdb_byte *syms_end = module_stream + mod_info->sym_byte_size;
  pdb_sym_parser parser (pdb, cu, func_ranges, lang,
			 sym_lines);
  scoped_restore restore_parser
    = make_scoped_restore (&pdb->cur_parser, &parser);

  /* We iterate by grabbing two values at a time: reclen and rectype.  In each
     loop check if there are 2 values available.  */
  gdb_byte *data = syms_start;
  while (data + PDB_RECORD_HDR_SIZE <= syms_end)
    {
      QUIT;
      auto hdr = pdb_parse_sym_record_hdr (data, syms_end);
      if (!hdr)
	{
	  pdb_warning ("bad record header at offset %td, stopping",
		       (ptrdiff_t) (data - syms_start));
	  break;
	}
      uint16_t rectype = hdr->type;
      size_t rec_size = hdr->rec_size ();

      gdb_byte *rec_data = data + PDB_RECORD_DATA_OFFS;

      /* Reject records too small to hold their fixed fields, or whose name
	 runs past the record, before any handler reads them.  */
      size_t body_size = rec_size - PDB_RECORD_DATA_OFFS;
      if (!pdb_sym_body_valid (rectype, rec_data, body_size))
	{
	  data += rec_size;
	  continue;
	}

      switch (rectype)
	{
	case S_GPROC32:
	case S_LPROC32:
	case S_GPROC32_ID:
	case S_LPROC32_ID:
	  parser.handle_func_sym (rec_data, rec_size - PDB_RECORD_DATA_OFFS,
				  rectype);
	  break;

	case S_THUNK32:
	  parser.handle_thunk_sym ();
	  break;

	case S_END:
	case S_PROC_ID_END:
	  parser.handle_scope_end_sym ();
	  break;
	}

      data += rec_size;
    }

  /* Close any remaining open scopes.  */
  while (!parser.scope_stack ().empty ())
    {
      CORE_ADDR end_addr = parser.scope_stack ().back ().end;
      parser.scope_stack ().pop_back ();
      cu->pop_context (end_addr, nullptr, end_addr != 0);
    }
}

/* See pdb-internal.h.  */

void
pdb_read_sym_record_stream (pdb_per_objfile *pdb)
{
  uint16_t idx = pdb->sym_record_stream;
  if (idx == 0 || idx == 0xFFFF)
    return;

  /* Move into the owned buffer — GSI lookups reference these bytes
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
  uint32_t hr_bytes = read_u32 (data + 8);
  uint32_t bucket_bytes = read_u32 (data + 12);

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

void
pdb_build_minsyms (pdb_per_objfile *pdb)
{
  if (pdb->sym_record_data.empty ())
    return;

  if (pdb->psgsi_stream == 0 || pdb->psgsi_stream == 0xFFFF)
    return;

  auto minsym_start = std::chrono::steady_clock::now ();

  /* PSGSI stream — used only here, auto-freed at scope exit.  */
  auto data_buf = pdb->read_stream (pdb->psgsi_stream);
  if (data_buf.size () < PSGSI_HDR_SIZE)
    return;

  gdb_byte *data = data_buf.data ();

  gdb_byte *gsi_data = data + PSGSI_HDR_SIZE;

  auto stream_size = pdb->streams[pdb->psgsi_stream].size;
  uint32_t sym_hash_size = read_u32 (data + PSGSI_HDR_SYM_HASH_OFFS);

  if (auto gsi_avail = stream_size - PSGSI_HDR_SIZE; sym_hash_size > gsi_avail)
    return;

  auto gsi = pdb_parse_gsi_hash_header (gsi_data, sym_hash_size);
  if (!gsi)
    return;

  minimal_symbol_reader reader (pdb->objfile);

  /* Natural names of data objects, keyed by unrelocated address.  A public
     carries only the decorated name and there is no MSVC demangler, so a
     data object that has both is better shown by this one.  */
  std::unordered_map<CORE_ADDR, std::pair<const char *, uint16_t>> data_names;
  {
    gdb_byte *rec = pdb->sym_record_data.data ();
    const gdb_byte *rec_end = rec + pdb->sym_record_data.size ();

    while (rec + PDB_RECORD_HDR_SIZE <= rec_end)
      {
	auto hdr = pdb_parse_sym_record_hdr (rec, rec_end);
	if (!hdr)
	  break;

	size_t rec_size = hdr->rec_size ();
	gdb_byte *body = rec + PDB_RECORD_DATA_OFFS;

	if ((hdr->type == S_GDATA32 || hdr->type == S_LDATA32)
	    && pdb_sym_body_valid (hdr->type, body,
				   rec_size - PDB_RECORD_DATA_OFFS))
	  {
	    uint32_t sect_offs = read_u32 (body
					   + PDB_SYMBOL_VAR_SECTION_OFFS_OFFS);
	    uint16_t sect_num = read_u16 (body
					  + PDB_SYMBOL_VAR_SECTION_NUM_OFFS);
	    const char *dname
	      = pdb_extract_string (body + PDB_SYMBOL_VAR_NAME_OFFS,
				    rec + rec_size);

	    if (pdb->section_valid (sect_num) && dname != nullptr
		&& *dname != '\0')
	      data_names.emplace
		(pdb->map_section_offset_unrelocated (sect_num, sect_offs),
		 std::make_pair (dname, sect_num));
	  }

	rec += rec_size;
      }
  }

  /* Walk GSI hash records (microsoft-pdb gsi.h: HRFile).
     Each record is GSI_HASH_RECORD_SIZE (8) bytes:
       uint32_t offs — SymRecordStream byte offset, stored as offset+1
		       so that 0 can mean "empty slot"
       uint32_t cref — reference count (unused here)  */
  for (uint32_t i = 0; i < gsi->hr.size () / GSI_HASH_RECORD_SIZE; i++)
    {
      const gdb_byte *rec = gsi->hr.data () + i * GSI_HASH_RECORD_SIZE;
      uint32_t offs = read_u32 (rec + GSI_HASH_RECORD_SYMOFFS_OFFS);

      /* Skip empty slots.  */
      if (offs == 0)
	continue;

      offs -= 1;
      if (offs > pdb->sym_record_data.size ()
	  || pdb->sym_record_data.size () - offs < PDB_RECORD_HDR_SIZE)
	{
	  pdb_warning ("PSGSI hash record %u: offset %u out of range "
		       "(sym_record_size=%zu)",
		       i, offs, pdb->sym_record_data.size ());
	  continue;
	}

      gdb_byte *sym_start = pdb->sym_record_data.data () + offs;
      const gdb_byte *syms_end = pdb->sym_record_data.data ()
				 + pdb->sym_record_data.size ();

      /* Validate the full record length fits within the symbol stream.  */
      auto hdr = pdb_parse_sym_record_hdr (sym_start, syms_end);
      if (!hdr)
	{
	  pdb_warning ("PSGSI hash record %u: bad record header at offset %u",
		       i, offs);
	  continue;
	}
      uint16_t reclen = hdr->len;
      uint16_t rectype = hdr->type;
      size_t rec_size = hdr->rec_size ();

      if (rectype != S_PUB32)
	continue;

      /* S_PUB32 body must contain at least flags + sect_offs + sect_num
	 + one byte for the (possibly empty) NUL-terminated name.  */
      const gdb_byte *rec_data = sym_start + PDB_RECORD_DATA_OFFS;
      const gdb_byte *rec_end = sym_start + rec_size;
      if (rec_data + PDB_SYMBOL_PUB_NAME_OFFS + 1 > rec_end)
	{
	  pdb_warning ("PSGSI hash record %u: S_PUB32 body truncated"
		       " (reclen=%u)",
		       i, reclen);
	  continue;
	}

      auto pub_flags = read_u32 (rec_data + PDB_SYMBOL_PUB_FLAGS_OFFS);
      auto sect_offs = read_u32 (rec_data + PDB_SYMBOL_PUB_SECT_OFFS_OFFS);
      auto sect_num = read_u16 (rec_data + PDB_SYMBOL_PUB_SECT_NUM_OFFS);
      const gdb_byte *name_ptr = rec_data + PDB_SYMBOL_PUB_NAME_OFFS;
      auto name = pdb_extract_string (name_ptr, rec_end);
      if (name == nullptr)
	{
	  pdb_warning ("PSGSI hash record %u: S_PUB32 name not"
		       " NUL-terminated",
		       i);
	  continue;
	}

      if (!pdb->section_valid (sect_num))
	continue;

      /* GDB adds the section offsets to a minimal symbol itself, and a DLL's
	 are already set when its symbols are read.  */
      CORE_ADDR unrel = pdb->map_section_offset_unrelocated (sect_num,
							     sect_offs);

      /* "??_C@" names an MSVC string literal, which ELF keeps as a local
	 .LC label outside the symbol table; as a minimal symbol it would
	 label every pointer to a literal ("<??_C@...> \"text\"").  */
      if (startswith (name, "??_C@"))
	continue;

      pdb->pub_mangled_names[unrel] = name;
      const bool decorated = name[0] == '?';

      /* cvpsfFunction.  */
      const bool is_function = (pub_flags & CV_PUBSYMFLAGS_FUNCTION) != 0;
      enum minimal_symbol_type type = is_function ? mst_text : mst_data;
      const char *msym_name = name;

      if (!is_function)
	{
	  auto nat = data_names.find (unrel);
	  if (nat != data_names.end () && decorated)
	    msym_name = nat->second.first;
	}

      int sect_idx = sect_num - 1;
      reader.record_with_info (msym_name, unrelocated_addr (unrel), type,
			       sect_idx);
    }

  double minsym_ms
    = std::chrono::duration<double, std::milli>
	(std::chrono::steady_clock::now () - minsym_start).count ();
  pdb_dbg_printf ("Built %zu minimal symbols in %.2f ms",
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

} /* namespace pdb */
