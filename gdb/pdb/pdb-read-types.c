/* PDB type reader.

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

/* TPI (Type Program Information) stream reader: builds the GDB types
   needed for function symbols.

   References:
     - microsoft-pdb: https://github.com/microsoft/microsoft-pdb
     - LLVM docs: https://llvm.org/docs/PDB/TpiStream.html

   TPI (stream 2) describes types, their members and function signatures.

   Records have consecutive indices and variable lengths.  We scan the
   stream once to build an array of record descriptors containing the kind,
   payload pointer and length.  A record can then be found directly through
   types[ti - type_idx_begin].

   GDB types are constructed as needed by resolving their referenced TIs.
   A separate cache retains the resulting type objects for reuse.

   Unsupported leaves resolve to TYPE_CODE_ERROR named
   "<unsupported PDB type>".  */

#include "symtab.h"
#include "gdbtypes.h"
#include "gdbarch.h"
#include "objfiles.h"
#include "complaints.h"
#include "typeprint.h"
#include "c-lang.h"
#include "cp-support.h"
#include "pdb/pdb-internal.h"
#include <string.h>
#include <algorithm>
#include <chrono>
#include "gdbsupport/selftest.h"

namespace pdb
{

/* Raw TPI record access and decoding.  */

/* Check that a type record has at least MIN_SIZE bytes of data.
   Returns true if OK.  Emits a warning and returns false if truncated.  */

static bool
pdb_check_record_size (const pdb_tpi_type *rec, uint32_t min_size,
		       const char *name)
{
  if (rec->data_len >= min_size)
    return true;
  pdb_complaint ("%s record truncated (got %u)", name, rec->data_len);
  return false;
}

/* Return the raw TPI record at TYPE_IDX, or nullptr if the record array
   is unavailable or the index is outside its range.  */

static const pdb_tpi_type *
pdb_tpi_get_type (const pdb_tpi_context *tpi, uint32_t type_idx)
{
  if (tpi->types == nullptr)
    return nullptr;

  if (type_idx < tpi->type_idx_begin || type_idx >= tpi->type_idx_end)
    return nullptr;

  uint32_t idx = type_idx - tpi->type_idx_begin;
  return &tpi->types[idx];
}

/* TPI stream loading.  */

/* TPI stream.
   https://llvm.org/docs/PDB/TpiStream.html

   The TPI stream (stream 2) holds the type records that describe the types
   the program uses.  Symbols reference types
   through a 32-bit Type Index (TI).  Type indices < TypeIndexBegin
   (0x1000) are built-in types whose meaning is encoded directly in the
   index value:
     bits  0-7  : type kind (void, int, float, ...)
     bits  8-10 : type mode (direct, near ptr, far ptr, ...); bit 11 is
		  reserved
   Any index >= TypeIndexBegin refers to a record in the TPI type record
   array.  Records form a topologically sorted DAG: record B may only
   reference record A if A's type index < B's type index.
   TPI stream layout:
     TpiStreamHeader  (56 bytes)
     Array of Type Records:
       Type Record Layout:
	  RecordLen (2 bytes)  : length of RecordKind + variable data
	  RecordKind (2 bytes) : Leaf type (LF_*)
	  RecordData (RecordLen - 2 bytes) : fields depend on RecordKind.  */

/* TPI/IPI header fields used during indexing.  */

/* Version.  */
inline constexpr auto TPI_HDR_VERSION_OFFS = 0;

/* HeaderSize.  */
inline constexpr auto TPI_HDR_HEADER_SIZE_OFFS = 4;
/* TypeIndexBegin.  */
inline constexpr auto TPI_HDR_TYPE_INDEX_BEGIN_OFFS = 8;
/* TypeIndexEnd.  */
inline constexpr auto TPI_HDR_TYPE_INDEX_END_OFFS = 12;
/* TypeRecordBytes.  */
inline constexpr auto TPI_HDR_TYPE_REC_BYTES_OFFS = 16;
/* Total header size.  */
inline constexpr auto TPI_HDR_SIZE = 56;

/* Header version accepted by this reader.  */
inline constexpr auto TPI_VERSION_V80 = 20040203;

/* Index the TPI stream: read the REC_BYTES bytes of records at
   REC_START_OFF and fill pdb->tpi.types with one entry per record,
   holding its leaf kind and a pointer to its data.  */

static void
pdb_parse_tpi_records (pdb_per_objfile *pdb, uint32_t rec_start_off,
		       uint32_t rec_bytes)
{
  pdb_tpi_context &tpi = pdb->tpi;

  /* Record indices start right after the simple types at 0x1000, and
     every record takes at least its header, so a header promising more
     records than its bytes can hold is corrupt.  Check before allocating
     for them.  */
  if (tpi.type_idx_begin != 0x1000 || tpi.type_idx_end < tpi.type_idx_begin
      || tpi.type_idx_end - tpi.type_idx_begin > rec_bytes / CV_REC_HDR_SIZE)
    pdb_error ("TPI: invalid type index range [0x%x, 0x%x) for %u bytes",
	       tpi.type_idx_begin, tpi.type_idx_end, rec_bytes);

  uint32_t num_types = tpi.type_idx_end - tpi.type_idx_begin;

  using clock = std::chrono::steady_clock;
  auto start = clock::now ();

  pdb_tpi_type *types = nullptr;
  if (num_types > 0)
    types = OBSTACK_CALLOC (&pdb->objfile->objfile_obstack, num_types,
			   pdb_tpi_type);

  gdb::byte_vector scratch;
  uint32_t off = rec_start_off;
  uint32_t rec_area_end = rec_start_off + rec_bytes;
  uint32_t idx = 0;

  while (idx < num_types)
    {
      if (off + CV_REC_HDR_SIZE > rec_area_end)
	pdb_error ("TPI: unexpected end of records at type 0x%x",
		   tpi.type_idx_begin + idx);

      /* RecordLen counts every byte after itself (RecordKind + data);
	 read it first to learn the record's total size.  */
      const gdb_byte *h = pdb_stream_bytes (pdb, PDB_STREAM_TPI, off,
					    CV_REC_HDR_SIZE, scratch);
      uint16_t rec_len = read_u16 (h);
      uint16_t rec_kind = read_u16 (h + 2);

      if (rec_len < 2)
	pdb_error ("TPI: type 0x%x has invalid RecordLen %u",
		   tpi.type_idx_begin + idx, rec_len);

      /* Full record = RecordLen field (2 bytes) + rec_len bytes after it.  */
      uint32_t rec_size = rec_len + 2;
      if (off + rec_size > rec_area_end)
	pdb_error ("TPI: type 0x%x overflows record area",
		   tpi.type_idx_begin + idx);

      /* Data is what follows RecordKind, so drop its 2 bytes from rec_len.  */
      uint32_t data_len = rec_len - 2;
      const gdb_byte *data
	= pdb_stream_bytes (pdb, PDB_STREAM_TPI, off + CV_REC_HDR_SIZE,
			    data_len, scratch);

      /* Scratch-backed payloads must be copied before the buffer is reused.
	 Mapping-backed payloads already have the objfile's lifetime.  */
      if (data == scratch.data ())
	data = (const gdb_byte *) obstack_copy (&pdb->objfile->objfile_obstack,
						data, data_len);

      types[idx].leaf = rec_kind;
      types[idx].data = data;
      types[idx].data_len = data_len;

      idx++;
      off += rec_size;
    }

  tpi.types = types;

  double ms = std::chrono::duration<double, std::milli>
    (clock::now () - start).count ();
  pdb_dbg_printf ("TPI: indexed %u type records in %.2f ms", num_types,
		  ms);
}

/* NAME with every anonymous-namespace scope spelled the way GDB spells
   it, "(anonymous namespace)", or nothing when NAME has none.  clang-cl
   writes "`anonymous namespace'" and cl "`anonymous-namespace'".  */

static std::optional<std::string>
pdb_gdb_anon_spelling (std::string_view name)
{
  static const std::string_view anon[] = {
    "`anonymous namespace'",
    "`anonymous-namespace'",
  };

  std::optional<std::string> out;
  size_t copied = 0;
  for (size_t i = name.find ('`'); i != std::string_view::npos;
       i = name.find ('`', i + 1))
    for (std::string_view a : anon)
      if (name.substr (i, a.size ()) == a)
	{
	  if (!out.has_value ())
	    out.emplace ();
	  out->append (name.substr (copied, i - copied));
	  out->append (CP_ANONYMOUS_NAMESPACE_STR);
	  copied = i + a.size ();
	  i = copied - 1;
	  break;
	}

  if (out.has_value ())
    out->append (name.substr (copied));
  return out;
}

/* See pdb-internal.h.  */

const char *
pdb_canonical_name (pdb_per_objfile *pdb, const char *name)
{
  if (name == nullptr)
    return nullptr;

  std::optional<std::string> anon = pdb_gdb_anon_spelling (name);
  if (anon.has_value ())
    name = anon->c_str ();

  /* A parameter list printed by c_type_print_args is not canonical either:
     a lookup of "f(const int &)" is canonicalized to "f(int const&)", which
     only a canonical symbol name matches, as dwarf2_canonicalize_name
     makes DWARF's.  */
  gdb::unique_xmalloc_ptr<char> canon = cp_canonicalize_string (name);
  if (canon == nullptr)
    return (anon.has_value ()
	    ? obstack_strdup (&pdb->objfile->objfile_obstack, *anon)
	    : name);

  return obstack_strdup (&pdb->objfile->objfile_obstack, canon.get ());
}

/* Index TPI (stream 2) and allocate the type cache.  Missing streams and
   detected format errors abort loading.  */

void
pdb_read_tpi_stream (pdb_per_objfile *pdb)
{
  auto &tpi = pdb->tpi;

  if (pdb->streams.size () <= PDB_STREAM_TPI
      || pdb->streams[PDB_STREAM_TPI].size == 0)
    pdb_error ("TPI stream (index 2) missing or empty");

  size_t tpi_size = pdb->streams[PDB_STREAM_TPI].size;
  if (tpi_size < TPI_HDR_SIZE)
    pdb_error ("TPI stream too small (%zu bytes)", tpi_size);

  gdb::byte_vector hdr_scratch;
  const gdb_byte *tpi_p = pdb_stream_bytes (pdb, PDB_STREAM_TPI, 0,
					    TPI_HDR_SIZE, hdr_scratch);

  uint32_t version = read_u32 (tpi_p + TPI_HDR_VERSION_OFFS);
  uint32_t hdr_size = read_u32 (tpi_p + TPI_HDR_HEADER_SIZE_OFFS);
  uint32_t rec_bytes = read_u32 (tpi_p + TPI_HDR_TYPE_REC_BYTES_OFFS);
  tpi.type_idx_begin = read_u32 (tpi_p + TPI_HDR_TYPE_INDEX_BEGIN_OFFS);
  tpi.type_idx_end = read_u32 (tpi_p + TPI_HDR_TYPE_INDEX_END_OFFS);

  if (version != TPI_VERSION_V80)
    pdb_error ("TPI unknown version 0x%08x (expected 0x%08x)", version,
	       TPI_VERSION_V80);

  if (hdr_size < TPI_HDR_SIZE)
    pdb_error ("TPI header size %u too small", hdr_size);

  if ((uint64_t) hdr_size + rec_bytes > tpi_size)
    pdb_error ("TPI records overflow stream (%u + %u > %zu)", hdr_size,
	       rec_bytes, tpi_size);

  uint32_t num_types = tpi.type_idx_end - tpi.type_idx_begin;
  pdb_dbg_printf ("TPI: TI=[0x%x..0x%x)  num_types=%u  rec_bytes=%u",
		  tpi.type_idx_begin, tpi.type_idx_end, num_types, rec_bytes);

  pdb_parse_tpi_records (pdb, hdr_size, rec_bytes);

  /* Cover simple indices [0, 0x1000) and record indices below
     type_idx_end.  */
  uint32_t cache_size = tpi.type_idx_end > 0x1000 ? tpi.type_idx_end : 0x1000;
  pdb->tpi.type_cache = OBSTACK_CALLOC (&pdb->objfile->objfile_obstack,
					cache_size, type *);
}

/* GDB type construction.  */

/* Return the per-objfile TYPE_CODE_ERROR placeholder for unsupported or
   unresolved PDB types, allocating it on first use.  */

static type &
pdb_tpi_get_unsupported_type (pdb_per_objfile *pdb)
{
  if (pdb->tpi.undefined_type != nullptr)
    return *pdb->tpi.undefined_type;

  /* Create an error type for unsupported/unresolved PDB types.  */
  type_allocator alloc (pdb->objfile, language_c);
  type *undef = alloc.new_type (TYPE_CODE_ERROR, 8, nullptr);
  undef->set_name ("<unsupported PDB type>");

  pdb->tpi.undefined_type = undef;
  return *undef;
}

/* Resolve a built-in type index to a GDB type.
   Simple type indices are < 0x1000 and encode:
     bits 0-7  : type kind
     bits 8-10 : pointer mode (bit 11 is reserved).
   Direct types use GDB builtins.  Pointer results and failures are cached
   by TI.  */

static type &
pdb_tpi_resolve_simple_type (pdb_per_objfile *pdb, uint32_t type_idx)
{
  /* pdb_read_tpi_stream allocates at least 0x1000 cache entries.  */
  type *cached = pdb->tpi.type_cache[type_idx];
  if (cached != nullptr)
    return *cached;

  gdbarch *garch = pdb->objfile->arch ();
  const struct builtin_type *bt = builtin_type (garch);

  uint32_t kind = cv_simple_kind (type_idx);
  uint32_t mode = cv_simple_mode (type_idx);

  /* First resolve the base type from the Kind field.  */
  type *base = nullptr;

  switch (kind)
    {
    /* Adopt void as a fallback if no type.  */
    case CV_NONE:
    case CV_VOID:
      base = bt->builtin_void;
      break;

    case CV_SIGNED_CHAR:
      base = bt->builtin_signed_char;
      break;

    case CV_UNSIGNED_CHAR:
      base = bt->builtin_unsigned_char;
      break;

    case CV_NARROW_CHAR:
      base = bt->builtin_char;
      break;

    case CV_WIDE_CHAR:
      base = bt->builtin_wchar;
      break;

    case CV_CHAR16:
      base = bt->builtin_char16;
      break;

    case CV_CHAR32:
      base = bt->builtin_char32;
      break;

    case CV_CHAR8:
      /* Mapped to unsigned char.  Not pedantically correct,
	 but good enough for the known use cases.  */
      base = bt->builtin_unsigned_char;
      break;

    case CV_INT8:
      base = bt->builtin_signed_char;
      break;

    case CV_UINT8:
      base = bt->builtin_unsigned_char;
      break;

    case CV_SHORT:
    case CV_INT16:
      base = bt->builtin_short;
      break;

    case CV_USHORT:
    case CV_UINT16:
      base = bt->builtin_unsigned_short;
      break;

    case CV_INT32:
      base = bt->builtin_int;
      break;

    case CV_UINT32:
      base = bt->builtin_unsigned_int;
      break;

    case CV_LONG:
      base = bt->builtin_long;
      break;

    case CV_ULONG:
      base = bt->builtin_unsigned_long;
      break;

    case CV_QUAD:
    case CV_INT64:
      base = bt->builtin_long_long;
      break;

    case CV_UQUAD:
    case CV_UINT64:
      base = bt->builtin_unsigned_long_long;
      break;

    case CV_OCT:
    case CV_INT128:
    case CV_UOCT:
    case CV_UINT128:
      {
	/* GDB's 128-bit builtins are named int128_t; the compiler's DWARF
	   names the type as the source does.  */
	bool is_unsigned = kind == CV_UOCT || kind == CV_UINT128;
	uint32_t direct = kind;
	if (pdb->tpi.type_cache[direct] == nullptr)
	  {
	    type_allocator alloc (pdb->objfile, language_c);
	    pdb->tpi.type_cache[direct]
	      = init_integer_type (alloc, 128, is_unsigned,
				   is_unsigned ? "unsigned __int128"
					       : "__int128");
	  }
	base = pdb->tpi.type_cache[direct];
      }
      break;

    case CV_FLOAT32:
      base = bt->builtin_float;
      break;

    case CV_FLOAT64:
      base = bt->builtin_double;
      break;

    case CV_FLOAT80:
      base = bt->builtin_long_double;
      break;

    case CV_COMPLEX32:
    case CV_COMPLEX64:
    case CV_COMPLEX80:
      {
	/* A complex primitive names its component's width, not its own.  */
	type *part = (kind == CV_COMPLEX32 ? bt->builtin_float
		      : kind == CV_COMPLEX64 ? bt->builtin_double
					     : bt->builtin_long_double);
	base = init_complex_type (nullptr, part);
      }
      break;

    case CV_BOOL8:
      base = bt->builtin_bool;
      break;

    case CV_HRESULT:
      base = bt->builtin_int;
      break;

    default:
      {
	pdb_complaint ("unknown simple type 0x%02x in TI 0x%04x", kind,
		       type_idx);
	type &undef = pdb_tpi_get_unsupported_type (pdb);
	pdb->tpi.type_cache[type_idx] = &undef;
	return undef;
      }
    }

  if (mode == CV_TM_DIRECT)
    return *base;

  /* Reject explicitly sized modes when they differ from the target's
     pointer size.  Substitution would give the wrong sizeof and stride.  */
  int arch_ptr_size = gdbarch_ptr_bit (pdb->objfile->arch ()) / 8;
  int encoded_size = 0;
  switch (mode)
    {
    case CV_TM_NPTR32:
    case CV_TM_FPTR32:
      encoded_size = 4;
      break;
    case CV_TM_NPTR64:
      encoded_size = 8;
      break;
    case CV_TM_NPTR128:
      encoded_size = 16;
      break;
    case CV_TM_NPTR:
    case CV_TM_FPTR:
    case CV_TM_HPTR:
      /* Legacy modes: no explicit size, assume arch.  */
      break;
    default:
      {
	/* Modes 8 to 15 are not defined.  */
	pdb_complaint ("simple type TI 0x%04x has unknown mode 0x%x",
		       type_idx, mode);
	type &undef = pdb_tpi_get_unsupported_type (pdb);
	pdb->tpi.type_cache[type_idx] = &undef;
	return undef;
      }
    }
  if (encoded_size != 0 && encoded_size != arch_ptr_size)
    {
      pdb_complaint ("simple-type pointer TI 0x%04x has mode 0x%x (size %d)"
		     " but arch pointer size is %d; marking unsupported",
		     type_idx, mode, encoded_size, arch_ptr_size);
      type &undef = pdb_tpi_get_unsupported_type (pdb);
      pdb->tpi.type_cache[type_idx] = &undef;
      return undef;
    }

  type *ptr_type = lookup_pointer_type (base);

  /* Cache the result before returning.  */
  pdb->tpi.type_cache[type_idx] = ptr_type;
  return *ptr_type;
}

/* Resolve the LF_ARGLIST record ARGLIST_TI into FUNC_TYPE's fields.  */

static void
pdb_tpi_resolve_arglist (pdb_per_objfile *pdb, type *func_type,
			 uint32_t arglist_ti)
{
  uint32_t argc = 0;
  auto *rec = pdb_tpi_get_type (&pdb->tpi, arglist_ti);

  if (rec == nullptr)
    pdb_complaint ("LF_ARGLIST: type index 0x%x out of range", arglist_ti);
  else if (rec->leaf != LF_ARGLIST)
    pdb_complaint ("LF_ARGLIST: type index 0x%x has unexpected leaf 0x%04x",
		   arglist_ti, rec->leaf);
  else if (pdb_check_record_size (rec, LF_ARGLIST_MIN_SIZE, "LF_ARGLIST"))
    {
      argc = read_u32 (rec->data + LF_ARGLIST_COUNT_OFFS);
      if (argc > (rec->data_len - LF_ARGLIST_ARGS_OFFS) / 4)
	{
	  pdb_complaint ("LF_ARGLIST: count %u exceeds record 0x%x", argc,
			 arglist_ti);
	  argc = 0;
	}
    }

  /* "..." is written as a trailing entry of type index 0, which is not a
     parameter and names no type.  */
  if (argc > 0
      && read_u32 (rec->data + LF_ARGLIST_ARGS_OFFS + (argc - 1) * 4) == 0)
    {
      func_type->set_has_varargs (true);
      argc--;
    }

  if (argc == 0)
    return;

  func_type->alloc_fields (argc);
  for (uint32_t i = 0; i < argc; i++)
    {
      auto arg_ti = read_u32 (rec->data + LF_ARGLIST_ARGS_OFFS + i * 4);
      auto *arg_type = &pdb_tpi_resolve_type (pdb, arg_ti);
      func_type->field (i).set_type (arg_type);
    }
}

/* Return a new function type returning RET_TYPE, with no parameters
   yet.  It is owned by the objfile, as its parameters can be: a builtin
   return type would make it owned by the architecture, and leave it
   pointing at freed types after an unload.  */

static type *
pdb_tpi_new_callable_type (pdb_per_objfile *pdb, type *ret_type)
{
  type *callable = type_allocator (pdb->objfile, language_c).new_type ();
  callable->set_code (TYPE_CODE_FUNC);
  callable->set_target_type (ret_type);
  callable->set_length (1);
  INIT_FUNC_SPECIFIC (callable);
  return callable;
}

/* LF_PROCEDURE.  */

static type &
pdb_tpi_make_procedure (pdb_per_objfile *pdb, const pdb_tpi_type *rec)
{
  if (!pdb_check_record_size (rec, LF_PROC_SIZE, "LF_PROCEDURE"))
    return pdb_tpi_get_unsupported_type (pdb);

  auto ret_ti = read_u32 (rec->data + LF_PROC_RVTYPE_OFFS);
  auto arglist_ti = read_u32 (rec->data + LF_PROC_ARGLIST_OFFS);

  auto *ret_type = &pdb_tpi_resolve_type (pdb, ret_ti);
  auto *func_type = pdb_tpi_new_callable_type (pdb, ret_type);

  /* Treat the recorded argument list as a prototype.  For TYPE_CODE_FUNC,
     infcall applies default promotions beyond the declared fields
     independently of is_prototyped, including a variadic tail.  */
  func_type->set_is_prototyped (true);

  pdb_tpi_resolve_arglist (pdb, func_type, arglist_ti);

  if (read_u8 (rec->data + LF_PROC_CALLTYPE_OFFS) == CV_CALL_NEAR_C)
    TYPE_CALLING_CONVENTION (func_type) = DW_CC_normal;
  return *func_type;
}

/* See pdb-internal.h.  */

type &
pdb_tpi_resolve_function_type (pdb_per_objfile *pdb, uint32_t type_idx)
{
  type &resolved = pdb_tpi_resolve_type (pdb, type_idx);
  if (resolved.code () == TYPE_CODE_FUNC)
    return resolved;

  if (type_idx != 0)
    pdb_complaint ("procedure type index 0x%x is not a procedure type",
		   type_idx);

  pdb_tpi_context &tpi = pdb->tpi;
  if (tpi.untyped_function_type == nullptr)
    {
      type *void_type = builtin_type (pdb->objfile->arch ())->builtin_void;
      tpi.untyped_function_type = pdb_tpi_new_callable_type (pdb, void_type);
    }
  return *tpi.untyped_function_type;
}

/* The longest parameter list printed into a function's name.
   Far above any real one, it stops a file whose types double in printed
   size at each level, as a function taking two pointers to the function
   one level down, from costing time and memory exponential in the depth,
   and bounds the memory the names take when many functions share one long
   parameter list.  */
inline constexpr size_t PDB_MAX_PARAMS_LENGTH = 16 * 1024;

/* The most parameters printed into a function's name.
   cp_canonicalize_string fails, warning each time, on a list of a few
   thousand.  */
inline constexpr int PDB_MAX_PARAMS = 512;

/* A string_file that stops the printing into it, by throwing, once it
   would hold more than PDB_MAX_PARAMS_LENGTH bytes.  */

class pdb_params_string_file : public string_file
{
public:
  void write (const char *buf, long length_buf) override
  {
    if (size () + length_buf > PDB_MAX_PARAMS_LENGTH)
      {
	m_overflow = true;
	error (_("parameter list too long"));
      }
    string_file::write (buf, length_buf);
  }

  bool overflow () const
  {
    return m_overflow;
  }

private:
  bool m_overflow = false;
};

/* Return the parameter list of FTYPE as c_type_print_args prints it in
   LANG, or nothing, with a complaint, when it has more than PDB_MAX_PARAMS
   parameters or is longer than PDB_MAX_PARAMS_LENGTH.  */

static std::optional<std::string>
pdb_print_params (type *ftype, enum language lang)
{
  if (ftype->num_fields () > PDB_MAX_PARAMS)
    {
      pdb_complaint ("parameter list of %d parameters not printed",
		     ftype->num_fields ());
      return {};
    }

  pdb_params_string_file buf;
  try
    {
      c_type_print_args (ftype, &buf, 1, lang, &type_print_raw_options);
    }
  catch (const gdb_exception_error &)
    {
      if (!buf.overflow ())
	throw;
      pdb_complaint ("parameter list longer than %zu bytes not printed",
		     PDB_MAX_PARAMS_LENGTH);
      return {};
    }
  return buf.release ();
}

/* See pdb-internal.h.  */

const char *
pdb_function_name (pdb_per_objfile *pdb, const char *name,
		   type *function_type, enum language lang)
{
  if (lang != language_cplus
      || (function_type->code () != TYPE_CODE_FUNC
	  && function_type->code () != TYPE_CODE_METHOD))
    return name;

  string_file buffer;
  buffer.puts (name);
  if (std::none_of (function_type->fields ().begin (),
		    function_type->fields ().end (),
		    [] (const field &item) { return !item.is_artificial (); })
      && !function_type->has_varargs ())
    buffer.puts ("()");
  else
    {
      auto printed = pdb->tpi.printed_param_lists.find (function_type);
      if (printed == pdb->tpi.printed_param_lists.end ())
	{
	  std::optional<std::string> params
	    = pdb_print_params (function_type, lang);
	  const char *list
	    = (params.has_value () ? pdb->objfile->intern (params->c_str ())
	       : nullptr);
	  printed = pdb->tpi.printed_param_lists.emplace (function_type,
							  list).first;
	}
      if (printed->second == nullptr)
	return name;
      buffer.puts (printed->second);
    }
  return pdb->objfile->intern
    (pdb_canonical_name (pdb, buffer.string ().c_str ()));
}

/* Build the GDB type for TPI record REC with the pdb_tpi_make_* helper
   of its leaf; non-type leaves resolve to
   builtin void and unknown leaves to the cached "<unsupported PDB type>".
   Never returns a null type.  */

static type &
pdb_tpi_build_type (pdb_per_objfile *pdb, const pdb_tpi_type *rec)
{
  switch (rec->leaf)
    {
    case LF_PROCEDURE:
      return pdb_tpi_make_procedure (pdb, rec);
    case LF_ARGLIST:
    case LF_FIELDLIST:
    case LF_VTSHAPE:
    case LF_LABEL:
    case LF_METHODLIST:
      return *builtin_type (pdb->objfile->arch ())->builtin_void;
    default:
      return pdb_tpi_get_unsupported_type (pdb);
    }
}

/* Recursive type resolver with caching: a record's type is built by
   pdb_tpi_build_type.  Unsupported leaf types return the cached
   "<unsupported PDB type>".  */

type &
pdb_tpi_resolve_type (pdb_per_objfile *pdb, uint32_t type_idx)
{
  /* Simple / built-in types.  */
  if (cv_ti_is_simple (type_idx))
    return pdb_tpi_resolve_simple_type (pdb, type_idx);

  pdb_tpi_context &tpi = pdb->tpi;

  /* A malformed record can reference a type index past the end of the TPI
     array.  Such an index has no record and no type_cache slot, so screen it
     out before indexing the cache to avoid an out-of-bounds access.  */
  if (type_idx >= tpi.type_idx_end)
    return pdb_tpi_get_unsupported_type (pdb);

  type *result = tpi.type_cache[type_idx];
  if (result == nullptr)
    {
      const pdb_tpi_type *rec = pdb_tpi_get_type (&tpi, type_idx);
      if (rec == nullptr)
	return pdb_tpi_get_unsupported_type (pdb);

      /* A record reached again while its build is in progress refers to
	 itself.  */
      if (std::find (tpi.types_in_progress.begin (),
		     tpi.types_in_progress.end (), type_idx)
	  != tpi.types_in_progress.end ())
	{
	  pdb_complaint ("type index 0x%x refers to itself", type_idx);
	  return pdb_tpi_get_unsupported_type (pdb);
	}

      /* Distinct records can each name the next, as a chain of procedure
	 types returning one another, so every build recurses one level.
	 Bound the depth so that such a chain cannot exhaust the stack.  The
	 types built above the cut are cached with it, so a type of such a
	 chain, reached first from far above it, stays cut short when it
	 is reached again directly.  Caching them is what keeps a chain
	 whose records each name the next twice from costing time
	 exponential in its depth.  */
      if (tpi.types_in_progress.size () >= 1000)
	{
	  pdb_complaint ("type index 0x%x nested too deeply", type_idx);
	  return pdb_tpi_get_unsupported_type (pdb);
	}

      tpi.types_in_progress.push_back (type_idx);
      auto pop = make_scope_exit ([&] ()
	{
	  tpi.types_in_progress.pop_back ();
	});

      result = &pdb_tpi_build_type (pdb, rec);
      tpi.type_cache[type_idx] = result;
    }

  return *result;
}

#if GDB_SELF_TEST

namespace selftests {

/* A TPI stream whose header is VERSION, HDR_SIZE, [BEGIN, END) and
   REC_BYTES, followed by zero bytes up to STREAM_SIZE.  */

static gdb::byte_vector
type_stream (uint32_t version, uint32_t hdr_size, uint32_t begin,
	     uint32_t end, uint32_t rec_bytes, size_t stream_size = 56 + 64)
{
  gdb::byte_vector s (TPI_HDR_SIZE, 0);
  put_u32 (s, TPI_HDR_VERSION_OFFS, version);
  put_u32 (s, TPI_HDR_HEADER_SIZE_OFFS, hdr_size);
  put_u32 (s, TPI_HDR_TYPE_INDEX_BEGIN_OFFS, begin);
  put_u32 (s, TPI_HDR_TYPE_INDEX_END_OFFS, end);
  put_u32 (s, TPI_HDR_TYPE_REC_BYTES_OFFS, rec_bytes);
  s.resize (stream_size);
  return s;
}

/* The TPI header: the record range and sizes are checked before anything
   is allocated for them, and a bad header drops the PDB.  */

static void
test_type_stream_headers ()
{
  /* Read STREAM as the TPI.  */
  auto read = [] (const gdb::byte_vector &stream)
    {
      test_pdb file;
      file.streams.resize (5);
      file.streams[PDB_STREAM_TPI] = stream;
      file.build ();
      test_pdb_reader r (file);
      r.read_msf ();
      return error_of ([&] { pdb_read_tpi_stream (&r.pdb); });
    };

  struct range_case { uint32_t begin, end; };
  /* An inverted range, more records than 64 bytes hold, one that
     overflows 32 bits, and a base other than 0x1000.  */
  const range_case ranges[] = { { 0x1000, 0xfff }, { 0x1000, 0x7fffffff },
				{ 0x1000, 0xffffffff }, { 0x2000, 0x2000 } };
  for (const range_case &rc : ranges)
    SELF_CHECK (read (type_stream (TPI_VERSION_V80, 56, rc.begin, rc.end, 64))
		.find ("TPI: invalid type index range") == 0);

  SELF_CHECK (read (type_stream (20040204, 56, 0x1000, 0x1000, 0))
	      .find ("TPI unknown version") == 0);
  SELF_CHECK (read (type_stream (TPI_VERSION_V80, 55, 0x1000, 0x1000, 0))
	      .find ("header size 55 too small") != std::string::npos);

  /* Records past the stream, also when the sum wraps 32 bits.  */
  for (uint32_t rec_bytes : { 65u, 0xffffffc8u })
    SELF_CHECK (read (type_stream (TPI_VERSION_V80, 56, 0x1000, 0x1000,
				   rec_bytes))
		.find ("TPI records overflow stream") == 0);

  SELF_CHECK (read (type_stream (TPI_VERSION_V80, 56, 0x1000, 0x1000, 0, 40))
	      .find ("TPI stream too small") == 0);
  SELF_CHECK (read (gdb::byte_vector ())
	      .find ("TPI stream (index 2) missing or empty") == 0);
}

/* Whether TYPE is the unsupported-type placeholder.  */

static bool
unsupported (const type *t)
{
  return t->name () != nullptr
	 && strcmp (t->name (), "<unsupported PDB type>") == 0;
}

/* Procedure types the reader must stop on: one returning itself, one
   returning a simple type of an undefined pointer mode, and a chain of
   20000 distinct procedure types, each returning the next, which is cut
   at a bounded depth.  A procedure's type index that names no procedure
   type gives a function returning void, as GDB requires a function to
   have a function type.  */

static void
test_procedure_types ()
{
  test_tpi tpi;
  uint32_t none = tpi.arglist ({});
  uint32_t self = tpi.procedure (tpi.next, none, 0);
  uint32_t badmode = tpi.procedure (0x874, none, 0);
  uint32_t deep = tpi.next;
  for (int i = 0; i < 20000; i++)
    tpi.procedure (i + 1 < 20000 ? tpi.next + 1 : 0x74, none, 0);

  test_read_modules ({}, {}, tpi, nullptr, [&] (pdb_per_objfile *pdb)
    {
      for (uint32_t ti : { self, badmode })
	{
	  type &t = pdb_tpi_resolve_type (pdb, ti);
	  SELF_CHECK (t.code () == TYPE_CODE_FUNC && t.num_fields () == 0
		      && unsupported (t.target_type ()));
	}

      const type *t = &pdb_tpi_resolve_type (pdb, deep);
      int depth = 0;
      while (t->code () == TYPE_CODE_FUNC)
	{
	  t = t->target_type ();
	  depth++;
	}
      SELF_CHECK (unsupported (t) && depth > 1 && depth < 20000);

      /* No type, a simple type and an argument list.  */
      for (uint32_t ti : { 0u, 0x74u, none })
	{
	  type &f = pdb_tpi_resolve_function_type (pdb, ti);
	  SELF_CHECK (f.code () == TYPE_CODE_FUNC && !f.is_prototyped ()
		      && f.target_type ()->code () == TYPE_CODE_VOID);
	}
    });
}

/* A procedure type belongs to the objfile even when it returns a builtin
   type, which belongs to the architecture, and so does the untyped
   function type: a copy Python keeps of one then survives the objfile's
   unload, as GDB preserves the objfile's types.  */

static void
test_procedure_owner ()
{
  test_tpi tpi;
  uint32_t f = tpi.procedure (0x74, tpi.arglist ({}), 0);

  test_read_modules ({}, {}, tpi, nullptr, [&] (pdb_per_objfile *pdb)
    {
      for (type *t : { &pdb_tpi_resolve_type (pdb, f),
		       &pdb_tpi_resolve_function_type (pdb, 0) })
	SELF_CHECK (t->code () == TYPE_CODE_FUNC && t->is_objfile_owned ()
		    && t->objfile_owner () == pdb->objfile);
    });
}

/* A C++ function's name has its parameter list, unless printing it would
   take too long: more parameters than PDB_MAX_PARAMS.  Such a function
   keeps its bare name, and its type's list is not tried again for the
   next function of that type.  */

static void
test_function_names ()
{
  test_tpi tpi;
  uint32_t one_int = tpi.procedure (0x74, tpi.arglist ({ 0x74 }), 1);
  uint32_t many = tpi.procedure (3, tpi.arglist (std::vector<uint32_t>
						  (2000, 0x74)), 2000);

  test_read_modules ({}, {}, tpi, nullptr, [&] (pdb_per_objfile *pdb)
    {
      auto name = [&] (const char *bare, uint32_t ti, enum language lang)
	{
	  return std::string (pdb_function_name
			      (pdb, bare, &pdb_tpi_resolve_type (pdb, ti),
			       lang));
	};
      SELF_CHECK (name ("f", one_int, language_cplus) == "f(int)");
      SELF_CHECK (name ("f", one_int, language_c) == "f");

      SELF_CHECK (name ("big", many, language_cplus) == "big");
      type *many_type = &pdb_tpi_resolve_type (pdb, many);
      auto printed = pdb->tpi.printed_param_lists.find (many_type);
      SELF_CHECK (printed != pdb->tpi.printed_param_lists.end ()
		  && printed->second == nullptr);
      SELF_CHECK (name ("big2", many, language_cplus) == "big2");
    });

  /* Parameter lists that double at each level: a procedure taking two
     procedures, each taking two, 14 levels down, prints far past
     PDB_MAX_PARAMS_LENGTH, so its function keeps its bare name.  */
  test_tpi doubling;
  uint32_t p = doubling.procedure (0x74, doubling.arglist ({}), 0);
  for (int k = 0; k < 14; k++)
    p = doubling.procedure (0x74, doubling.arglist ({ p, p }), 2);
  uint32_t f = doubling.procedure (3, doubling.arglist ({ p }), 1);

  test_read_modules ({}, {}, doubling, nullptr, [&] (pdb_per_objfile *pdb)
    {
      type *ftype = &pdb_tpi_resolve_type (pdb, f);
      SELF_CHECK (strcmp (pdb_function_name (pdb, "f", ftype, language_cplus),
			  "f") == 0);
      auto printed = pdb->tpi.printed_param_lists.find (ftype);
      SELF_CHECK (printed != pdb->tpi.printed_param_lists.end ()
		  && printed->second == nullptr);
    });
}

} /* namespace selftests */

#endif /* GDB_SELF_TEST */

} /* namespace pdb */

INIT_GDB_FILE (pdb_read_types)
{
#if GDB_SELF_TEST
  selftests::register_test ("pdb-procedure-types",
			    pdb::selftests::test_procedure_types);
  selftests::register_test ("pdb-procedure-owner",
			    pdb::selftests::test_procedure_owner);
  selftests::register_test ("pdb-function-names",
			    pdb::selftests::test_function_names);
  selftests::register_test ("pdb-type-stream-headers",
			    pdb::selftests::test_type_stream_headers);
#endif
}
