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

/* Decode an integer CodeView numeric leaf, used for sizes, member offsets,
   enumerators and constants.  Return the bytes consumed, including the tag,
   or 0 for truncation or an unsupported encoding.

   A leading uint16_t below 0x8000 is the value itself (2 bytes consumed).
   Otherwise it identifies the following payload: LF_CHAR has 1 byte,
   LF_SHORT/LF_USHORT 2, LF_LONG/LF_ULONG 4, and LF_QUADWORD/LF_UQUADWORD 8.
   Other numeric kinds, such as floating-point leaves, are not decoded here.

   Store the result in *VALUE.  Signed values are represented modulo 2^64;
   the caller supplies the interpretation and advances by the returned size
   before reading the next field.  */

uint32_t
pdb_cv_read_numeric (const gdb_byte *data, uint32_t max_len, uint64_t *value)
{
  if (max_len < 2)
    return 0;

  auto leaf = read_u16 (data);

  /* Simple leaf value - encoded directly in the leaf (2 bytes).  */
  if (leaf < LF_NUMERIC)
    {
      *value = leaf;
      return 2;
    }

  const gdb_byte *payload = data + 2;

  switch (leaf)
    {
    case LF_CHAR:
      if (max_len < 3)
	return 0;
      *value = read_i8 (payload);
      return 3;
    case LF_SHORT:
      if (max_len < 4)
	return 0;
      *value = read_i16 (payload);
      return 4;
    case LF_USHORT:
      if (max_len < 4)
	return 0;
      *value = read_u16 (payload);
      return 4;
    case LF_LONG:
      if (max_len < 6)
	return 0;
      *value = read_i32 (payload);
      return 6;
    case LF_ULONG:
      if (max_len < 6)
	return 0;
      *value = read_u32 (payload);
      return 6;
    case LF_QUADWORD:
    case LF_UQUADWORD:
      if (max_len < 10)
	return 0;
      *value = read_u64 (payload);
      return 10;
    default:
      pdb_complaint ("numeric leaf 0x%04x is not an integer", leaf);
      return 0;
    }
}

/* See pdb-internal.h.  */

const char *
pdb_cv_numeric_and_name (const pdb_tpi_type *rec, uint32_t offs,
			 uint64_t *value)
{
  if (offs >= rec->data_len)
    return nullptr;

  uint32_t nr = pdb_cv_read_numeric (rec->data + offs, rec->data_len - offs,
				     value);
  if (nr == 0)
    return nullptr;
  return pdb_extract_string (rec->data + offs + nr,
			     rec->data + rec->data_len);
}

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

  /* Reject a header whose index range has more records than its bytes
     can hold, before allocating for them.  Such a header is corrupt,
     because record indices start right after the simple types at 0x1000
     and every record takes at least its header.  */
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

      /* The full record is the 2-byte RecordLen field followed by the
	 rec_len bytes after it.  */
      uint32_t rec_size = rec_len + 2;
      if (off + rec_size > rec_area_end)
	pdb_error ("TPI: type 0x%x overflows record area",
		   tpi.type_idx_begin + idx);

      /* Drop the 2 bytes of RecordKind from rec_len, as the data is what
	 follows RecordKind.  */
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

  /* Canonicalize the name, as a parameter list printed by
     c_type_print_args is not canonical either.  A lookup of
     "f(const int &)" is canonicalized to "f(int const&)", which only a
     canonical symbol name matches, as dwarf2_canonicalize_name makes
     DWARF's.  */
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
      /* Map the type to unsigned char.  This is not pedantically correct,
	 but it is good enough for the known use cases.  */
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
	/* Name the 128-bit type as the source does, as the compiler's DWARF
	   does.  GDB's 128-bit builtins are named int128_t.  */
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
      /* Assume the architecture's pointer size for the legacy modes,
	 which have no explicit size.  */
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

/* LF_MODIFIER: apply the recorded const and volatile qualifiers.  */

static type &
pdb_tpi_make_modifier (pdb_per_objfile *pdb, const pdb_tpi_type *rec)
{
  if (!pdb_check_record_size (rec, LF_MOD_SIZE, "LF_MODIFIER"))
    return pdb_tpi_get_unsupported_type (pdb);

  auto type_idx = read_u32 (rec->data + LF_MOD_TYPE_OFFS);
  auto attr = read_u16 (rec->data + LF_MOD_ATTR_OFFS);

  auto *base = &pdb_tpi_resolve_type (pdb, type_idx);

  bool is_const = (attr & CV_MODIFIER_CONST) != 0;
  bool is_volatile = (attr & CV_MODIFIER_VOLATILE) != 0;
  if (is_const || is_volatile)
    base = make_cv_type (is_const || base->is_const (),
			is_volatile || base->is_volatile (), base);

  return *base;
}

/* LF_POINTER: construct a pointer, reference or member-pointer type.

   The attr's pointer type (CV_ptrtype_e) says how the address is stored.
   Only flat addresses are read: CV_PTR_NEAR32 (4 bytes) and CV_PTR_64
   (8 bytes), and only when the width is the target's.  16-bit segmented
   pointers (near, far, huge) and 16:32 far pointers are ignored; GDB does
   not support segmented addresses.  MSVC's __based pointers (an offset
   from a base pointer) and __ptr32 pointers on a 64-bit target are not
   supported.  All of these become the unsupported type.  A pointer or
   reference has the length GDB gives it, 8 bytes on x64.

   The size field in attr is checked only for a plain pointer, where it
   must equal the width; a mismatch is a complaint.  It is never read for
   a reference: for a parameter that x64 passes by address, clang-cl types
   the parameter as a reference and records the referenced type's size
   there (16 for a _Complex double, 0 for a typedef).

   Member-pointer lengths come from GDB's C++ ABI layer.  */

static type &
pdb_tpi_make_pointer (pdb_per_objfile *pdb, const pdb_tpi_type *rec)
{
  if (!pdb_check_record_size (rec, LF_POINTER_SIZE, "LF_POINTER"))
    return pdb_tpi_get_unsupported_type (pdb);

  uint32_t utype = read_u32 (rec->data + LF_POINTER_UTYPE_OFFS);
  uint32_t attr = read_u32 (rec->data + LF_POINTER_ATTR_OFFS);

  uint32_t ptrmode = cv_ptr_mode (attr);
  bool is_const = (attr & LF_POINTER_ATTR_ISCONST) != 0;
  bool is_volatile = (attr & LF_POINTER_ATTR_ISVOLATILE) != 0;
  bool is_restrict = (attr & LF_POINTER_ATTR_ISRESTRICT) != 0;

  uint32_t ptrtype = cv_ptr_type (attr);
  uint32_t width = cv_ptr_type_width (ptrtype);
  uint32_t arch_ptr_size = gdbarch_ptr_bit (pdb->objfile->arch ()) / 8;
  if (width != arch_ptr_size)
    {
      pdb_complaint ("LF_POINTER to type 0x%x has pointer type 0x%x, not a"
		     " %u-byte address", utype, ptrtype, arch_ptr_size);
      return pdb_tpi_get_unsupported_type (pdb);
    }

  if (ptrmode == CV_PTRMODE_POINTER)
    {
      uint32_t size = cv_ptr_size (attr);
      if (size != 0 && size != width)
	pdb_complaint ("LF_POINTER to type 0x%x records size %u for a %u-byte"
		       " pointer", utype, size, width);
    }

  auto *pointee = &pdb_tpi_resolve_type (pdb, utype);

  type *result;
  switch (ptrmode)
    {
    case CV_PTRMODE_LVALUE_REF:
      result = lookup_lvalue_reference_type (pointee);
      break;
    case CV_PTRMODE_RVALUE_REF:
      result = lookup_rvalue_reference_type (pointee);
      break;
    case CV_PTRMODE_PMFUNC:
      pdb_complaint ("LF_POINTER to member function 0x%x is not a method"
		     " type", utype);
      return pdb_tpi_get_unsupported_type (pdb);
    case CV_PTRMODE_PMEM:
      {
	/* Make a pointer to data member of the class that follows the fixed
	   part of the record.  The class must be there and must be a
	   class.  */
	if (rec->data_len < LF_POINTER_PMCLASS_OFFS + 4)
	  {
	    pdb_complaint ("LF_POINTER to member 0x%x has no class", utype);
	    return pdb_tpi_get_unsupported_type (pdb);
	  }

	uint32_t pmclass_ti = read_u32 (rec->data + LF_POINTER_PMCLASS_OFFS);
	type *domain = &pdb_tpi_resolve_type (pdb, pmclass_ti);
	if (domain->code () != TYPE_CODE_STRUCT
	    && domain->code () != TYPE_CODE_UNION)
	  {
	    pdb_complaint ("LF_POINTER to member 0x%x names class 0x%x, which"
			   " is not a class", utype, pmclass_ti);
	    return pdb_tpi_get_unsupported_type (pdb);
	  }
	result = lookup_memberptr_type (pointee, domain);
	break;
      }
    case CV_PTRMODE_POINTER:
    default:
      result = lookup_pointer_type (pointee);
      break;
    }

  if (is_const || is_volatile)
    result = make_cv_type (is_const, is_volatile, result);
  if (is_restrict)
    result = make_restrict_type (result);

  return *result;
}

/* LF_ARRAY.  */

static type &
pdb_tpi_make_array (pdb_per_objfile *pdb, const pdb_tpi_type *rec)
{
  if (!pdb_check_record_size (rec, LF_ARRAY_MIN_SIZE, "LF_ARRAY"))
    return pdb_tpi_get_unsupported_type (pdb);

  uint32_t elem_ti = read_u32 (rec->data + LF_ARRAY_ELEMTYPE_OFFS);
  uint32_t idx_ti = read_u32 (rec->data + LF_ARRAY_IDXTYPE_OFFS);

  auto *elem_type = &pdb_tpi_resolve_type (pdb, elem_ti);
  auto *idx_type = &pdb_tpi_resolve_type (pdb, idx_ti);

  /* Replace an index type that is not an integer with a size by unsigned
     long long.  A record can name void or an empty structure, and
     create_range_type asserts on both.  */
  if (!is_integral_type (idx_type) || idx_type->length () == 0)
    {
      pdb_complaint ("LF_ARRAY index type 0x%x is not an integer type",
		     idx_ti);
      idx_type
	= builtin_type (pdb->objfile->arch ())->builtin_unsigned_long_long;
    }

  /* Trailing data: numeric leaf for total size in bytes, then a
     NUL-terminated name (may be empty).  */
  const gdb_byte *d = rec->data + LF_ARRAY_DATA_OFFS;
  const gdb_byte *end = rec->data + rec->data_len;
  uint32_t data_len = rec->data_len - LF_ARRAY_DATA_OFFS;

  uint64_t total_size = 0;
  uint32_t consumed = 0;
  if (data_len > 0)
    consumed = pdb_cv_read_numeric (d, data_len, &total_size);

  if (consumed == 0 && data_len > 0)
    pdb_complaint ("LF_ARRAY: failed to read numeric leaf for array size");

  const char *name = nullptr;
  if (consumed > 0)
    name = pdb_extract_string (d + consumed, end);

  uint64_t elem_size = elem_type->length ();

  /* Member-pointer lengths come from GDB's C++ ABI rather than from the
     PDB, so count elements with the size the record itself declares.  */
  if (elem_type->code () == TYPE_CODE_METHODPTR
      || elem_type->code () == TYPE_CODE_MEMBERPTR)
    {
      /* A qualified element names the modifier, which names the pointer.  */
      uint32_t ptr_ti = elem_ti;
      const pdb_tpi_type *elem_rec = pdb_tpi_get_type (&pdb->tpi, ptr_ti);
      if (elem_rec != nullptr && elem_rec->leaf == LF_MODIFIER
	  && elem_rec->data_len >= LF_MOD_SIZE)
	{
	  ptr_ti = read_u32 (elem_rec->data + LF_MOD_TYPE_OFFS);
	  elem_rec = pdb_tpi_get_type (&pdb->tpi, ptr_ti);
	}

      if (elem_rec != nullptr && elem_rec->leaf == LF_POINTER
	  && elem_rec->data_len >= LF_POINTER_SIZE)
	{
	  uint32_t cv_size
	    = cv_ptr_size (read_u32 (elem_rec->data + LF_POINTER_ATTR_OFFS));
	  if (cv_size > 0)
	    elem_size = cv_size;
	}
    }

  uint64_t num_elements = 0;
  if (elem_size > 0)
    {
      num_elements = total_size / elem_size;
      if (total_size % elem_size != 0)
	pdb_complaint ("LF_ARRAY: total size %" PRIu64 " not a multiple of"
		       " element size %" PRIu64,
		       total_size, elem_size);
    }

  /* An undefined upper bound accommodates flexible members, but does not
     distinguish them from fixed zero-length arrays.  */
  LONGEST high_bound = num_elements > 0 ? (LONGEST) num_elements - 1 : 0;

  type_allocator alloc (pdb->objfile, language_c);
  auto *range_type = create_static_range_type (alloc, idx_type, 0, high_bound);
  if (num_elements == 0)
    range_type->bounds ()->high.set_undefined ();
  auto *array_type = create_array_type (alloc, elem_type, range_type);

  /* Reserve enough bytes for GDB to walk NUM_ELEMENTS using ELEM_TYPE's
     length, even if it exceeds the encoded element size.  This protects
     that walk's bounds; it does not correct the member-pointer layout.  */
  uint64_t walk_size = num_elements * (uint64_t) elem_type->length ();
  array_type->set_length (walk_size > total_size ? walk_size : total_size);

  if (name != nullptr && *name != '\0')
    array_type->set_name (name);
  return *array_type;
}

/* LF_BITFIELD.  */

static type &
pdb_tpi_make_bitfield (pdb_per_objfile *pdb, const pdb_tpi_type *rec)
{
  if (!pdb_check_record_size (rec, LF_BITFIELD_SIZE, "LF_BITFIELD"))
    return pdb_tpi_get_unsupported_type (pdb);

  auto base_ti = read_u32 (rec->data + LF_BITFIELD_TYPE_OFFS);

  return pdb_tpi_resolve_type (pdb, base_ti);
}

/* Return the underlying type of an enumeration, UTYPE_TI.  A malformed
   record can name a type that is not an integer; complain and use int
   instead.  */

static type *
pdb_enum_underlying_type (pdb_per_objfile *pdb, uint32_t utype_ti)
{
  type *utype = &pdb_tpi_resolve_type (pdb, utype_ti);
  if (is_integral_type (utype) && utype->length () > 0)
    return utype;

  pdb_complaint ("LF_ENUM underlying type 0x%x is not an integer type",
		 utype_ti);
  return builtin_type (pdb->objfile->arch ())->builtin_int;
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
  if (lang != language_cplus_
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

/* Until their members are read, a class, structure, union or enumeration
   is an incomplete type with the name and size of its record.  */

static type &
pdb_tpi_make_incomplete_tag (pdb_per_objfile *pdb, const pdb_tpi_type *rec)
{
  enum type_code code = TYPE_CODE_STRUCT;
  uint32_t min_size = LF_STRUCT_MIN_SIZE;
  uint32_t data_offs = LF_STRUCT_DATA_OFFS;
  const char *what = "LF_STRUCTURE/CLASS";
  if (rec->leaf == LF_UNION)
    {
      code = TYPE_CODE_UNION;
      min_size = LF_UNION_MIN_SIZE;
      data_offs = LF_UNION_DATA_OFFS;
      what = "LF_UNION";
    }
  else if (rec->leaf == LF_ENUM)
    {
      code = TYPE_CODE_ENUM;
      min_size = LF_ENUM_MIN_SIZE;
      data_offs = LF_ENUM_NAME_OFFS;
      what = "LF_ENUM";
    }

  if (!pdb_check_record_size (rec, min_size, what))
    return pdb_tpi_get_unsupported_type (pdb);

  type *underlying = nullptr;
  uint64_t size = 0;
  const char *name = nullptr;
  if (code == TYPE_CODE_ENUM)
    {
      uint32_t utype_ti = read_u32 (rec->data + LF_ENUM_UTYPE_OFFS);
      underlying = pdb_enum_underlying_type (pdb, utype_ti);
      size = underlying->length ();
      name = pdb_extract_string (rec->data + data_offs,
				 rec->data + rec->data_len);
    }
  else
    {
      name = pdb_cv_numeric_and_name (rec, data_offs, &size);
      if (name == nullptr)
	return pdb_tpi_get_unsupported_type (pdb);
    }

  type_allocator alloc (pdb->objfile, language_c);
  type *result = alloc.new_type ();
  if (code != TYPE_CODE_ENUM)
    INIT_CPLUS_SPECIFIC (result);
  result->set_code (code);
  result->set_length (size);
  result->set_is_stub (true);
  if (name != nullptr && name[0] != '\0')
    result->set_name (pdb_canonical_name (pdb, name));
  if (rec->leaf == LF_CLASS)
    result->set_is_declared_class (true);
  if (underlying != nullptr)
    {
      result->set_target_type (underlying);
      result->set_is_unsigned (underlying->is_unsigned ());
    }
  return *result;
}

/* Build the GDB type for TPI record REC with the pdb_tpi_make_* helper
   of its leaf.  A non-type leaf resolves to builtin void, and an unknown
   leaf to the cached "<unsupported PDB type>".  The result is never
   null.  */

static type &
pdb_tpi_build_type (pdb_per_objfile *pdb, const pdb_tpi_type *rec)
{
  switch (rec->leaf)
    {
    case LF_MODIFIER:
      return pdb_tpi_make_modifier (pdb, rec);
    case LF_PROCEDURE:
      return pdb_tpi_make_procedure (pdb, rec);
    case LF_POINTER:
      return pdb_tpi_make_pointer (pdb, rec);
    case LF_ARRAY:
      return pdb_tpi_make_array (pdb, rec);
    case LF_BITFIELD:
      return pdb_tpi_make_bitfield (pdb, rec);
    case LF_STRUCTURE:
    case LF_CLASS:
    case LF_UNION:
    case LF_ENUM:
      return pdb_tpi_make_incomplete_tag (pdb, rec);
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

  /* Reject a type index past the end of the TPI array before
     indexing the cache, to avoid an out-of-bounds access.  A malformed
     record can reference such an index, which has no record and no
     type_cache slot.  */
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

      /* Bound the depth so that a chain of distinct records, each naming
	 the next, cannot exhaust the stack.  In such a chain, as of
	 procedure types returning one another, every build recurses one
	 level.  The types built above the cut are cached with it, so a type
	 of such a chain, reached first from far above it, stays cut short
	 when it is reached again directly.  Caching them is what keeps a
	 chain whose records each name the next twice from costing time
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

/* Numeric leaves: a value below LF_NUMERIC is the leaf itself; above, the
   leaf names a little-endian value of 1, 2, 4 or 8 bytes after it.  A leaf
   cut short anywhere, or another kind, reads nothing.  */

static void
test_cv_read_numeric ()
{
  struct numeric_case
  {
    std::vector<gdb_byte> bytes;
    uint64_t value;
  };
  const numeric_case cases[] = {
    { { 0x00, 0x00 }, 0 },
    { { 0xff, 0x7f }, 0x7fff },
    /* LF_CHAR -5.  */
    { { 0x00, 0x80, 0xfb }, (uint64_t) -5 },
    /* LF_SHORT -2, LF_USHORT 0xfffe.  */
    { { 0x01, 0x80, 0xfe, 0xff }, (uint64_t) -2 },
    { { 0x02, 0x80, 0xfe, 0xff }, 0xfffe },
    /* LF_LONG -0x12345678, LF_ULONG 0x87654321.  */
    { { 0x03, 0x80, 0x88, 0xa9, 0xcb, 0xed }, (uint64_t) -0x12345678 },
    { { 0x04, 0x80, 0x21, 0x43, 0x65, 0x87 }, 0x87654321 },
    /* LF_QUADWORD -0x123456789.  */
    { { 0x09, 0x80, 0x77, 0x98, 0xba, 0xdc, 0xfe, 0xff, 0xff, 0xff },
      (uint64_t) -0x123456789 },
    /* What clang-cl writes for the enumerators 0x123456789ABCDEF0 of an
       unsigned long long enum and -0x123456789 of a long long one: both
       LF_UQUADWORD.  */
    { { 0x0a, 0x80, 0xf0, 0xde, 0xbc, 0x9a, 0x78, 0x56, 0x34, 0x12 },
      0x123456789abcdef0 },
    { { 0x0a, 0x80, 0x77, 0x98, 0xba, 0xdc, 0xfe, 0xff, 0xff, 0xff },
      (uint64_t) -0x123456789 },
  };

  for (const numeric_case &c : cases)
    {
      uint64_t value = 0;
      SELF_CHECK (pdb_cv_read_numeric (c.bytes.data (), c.bytes.size (),
				       &value) == c.bytes.size ());
      SELF_CHECK (value == c.value);

      /* Trailing bytes are not read.  */
      std::vector<gdb_byte> longer = c.bytes;
      longer.push_back (0xaa);
      SELF_CHECK (pdb_cv_read_numeric (longer.data (), longer.size (),
				       &value) == c.bytes.size ());

      for (size_t len = 0; len < c.bytes.size (); len++)
	SELF_CHECK (pdb_cv_read_numeric (c.bytes.data (), len, &value) == 0);
    }

  /* LF_REAL32, LF_COMPLEX32 and an unknown kind are not integers.  */
  for (uint16_t leaf : { 0x8005, 0x800c, 0x8fff })
    {
      gdb_byte bytes[16] = { (gdb_byte) leaf, (gdb_byte) (leaf >> 8) };
      uint64_t value;
      SELF_CHECK (pdb_cv_read_numeric (bytes, sizeof bytes, &value) == 0);
    }
}

/* Build the body of an LF_POINTER record to TARGET in pointer mode MODE.
   The pointer is a 64-bit, 8-byte pointer with the attribute bits EXTRA
   added.  When PMCLASS is not 0, the body ends with it as the class of a
   pointer to member.  */

static gdb::byte_vector
pointer_body (uint32_t target, uint32_t mode, uint32_t extra = 0,
	      uint32_t pmclass = 0)
{
  gdb::byte_vector body;
  put_u32 (body, 0, target);
  put_u32 (body, 4, (CV_PTRTYPE_64 | (mode << LF_POINTER_ATTR_PTRMODE_SHIFT)
		     | (8 << LF_POINTER_ATTR_SIZE_SHIFT) | extra));
  if (pmclass != 0)
    put_u32 (body, 8, pmclass);
  return body;
}

/* Build the body of an LF_STRUCTURE, LF_CLASS or LF_UNION record (LEAF)
   named NAME, of SIZE bytes, with the property bits PROPERTY.  The size
   is written as a direct numeric leaf.  */

static gdb::byte_vector
tag_body (uint16_t leaf, uint16_t size, const char *name,
	  uint16_t property = 0)
{
  gdb::byte_vector body;
  body.push_back (0);
  body.push_back (0);
  body.push_back (property & 0xff);
  body.push_back (property >> 8);
  put_u32 (body, 4, 0);
  if (leaf != LF_UNION)
    {
      put_u32 (body, 8, 0);
      put_u32 (body, 12, 0);
    }
  body.push_back (size & 0xff);
  body.push_back (size >> 8);
  body.insert (body.end (), name, name + strlen (name) + 1);
  return body;
}

/* Build the body of an LF_ENUM record named NAME whose underlying type
   is UTYPE.  */

static gdb::byte_vector
enum_body (uint32_t utype, const char *name)
{
  gdb::byte_vector body (4, 0);
  put_u32 (body, 4, utype);
  put_u32 (body, 8, 0);
  body.insert (body.end (), name, name + strlen (name) + 1);
  return body;
}

/* Build the body of an LF_ARRAY record named NAME, of SIZE bytes, whose
   elements are ELEM and whose index type is IDX.  */

static gdb::byte_vector
array_body (uint32_t elem, uint32_t idx, uint16_t size, const char *name)
{
  gdb::byte_vector body;
  put_u32 (body, 0, elem);
  put_u32 (body, 4, idx);
  body.push_back (size & 0xff);
  body.push_back (size >> 8);
  body.insert (body.end (), name, name + strlen (name) + 1);
  return body;
}

/* Test the types LF_POINTER records give.  Check a pointer, the two
   kinds of reference, the const, volatile and restrict qualifiers, and a
   pointer to data member of a class.  Check that the unsupported type
   comes from a 32-bit pointer on x64, a pointer to member whose class is
   missing or is not a class, a pointer to member function whose target
   is not a method, and a record cut short.  */

static void
test_pointer_types ()
{
  test_tpi tpi;
  uint32_t cls = tpi.add (LF_STRUCTURE, tag_body (LF_STRUCTURE, 4, "Cls"));
  uint32_t ptr = tpi.add (LF_POINTER, pointer_body (0x74, CV_PTRMODE_POINTER));
  uint32_t lref
    = tpi.add (LF_POINTER, pointer_body (0x74, CV_PTRMODE_LVALUE_REF));
  uint32_t rref
    = tpi.add (LF_POINTER, pointer_body (0x74, CV_PTRMODE_RVALUE_REF));
  uint32_t cvr = tpi.add (LF_POINTER,
			  pointer_body (0x74, CV_PTRMODE_POINTER,
					(LF_POINTER_ATTR_ISCONST
					 | LF_POINTER_ATTR_ISVOLATILE
					 | LF_POINTER_ATTR_ISRESTRICT)));
  gdb::byte_vector near32 = pointer_body (0x74, CV_PTRMODE_POINTER);
  put_u32 (near32, 4, CV_PTRTYPE_NEAR32 | (4 << LF_POINTER_ATTR_SIZE_SHIFT));
  uint32_t ptr32 = tpi.add (LF_POINTER, near32);
  uint32_t pmem
    = tpi.add (LF_POINTER, pointer_body (0x74, CV_PTRMODE_PMEM, 0, cls));
  uint32_t pmem_short
    = tpi.add (LF_POINTER, pointer_body (0x74, CV_PTRMODE_PMEM));
  uint32_t pmem_int
    = tpi.add (LF_POINTER, pointer_body (0x74, CV_PTRMODE_PMEM, 0, 0x74));
  uint32_t pmfunc
    = tpi.add (LF_POINTER, pointer_body (0x74, CV_PTRMODE_PMFUNC, 0, cls));
  uint32_t cut = tpi.add (LF_POINTER, gdb::byte_vector (4, 0));

  test_read_modules ({}, {}, tpi, nullptr, [&] (pdb_per_objfile *pdb)
    {
      auto resolve = [&] (uint32_t ti)
	{
	  return &pdb_tpi_resolve_type (pdb, ti);
	};

      type *t = resolve (ptr);
      SELF_CHECK (t->code () == TYPE_CODE_PTR && t->length () == 8
		  && t->target_type ()->code () == TYPE_CODE_INT);
      SELF_CHECK (resolve (lref)->code () == TYPE_CODE_REF);
      SELF_CHECK (resolve (rref)->code () == TYPE_CODE_RVALUE_REF);
      t = resolve (cvr);
      SELF_CHECK (t->code () == TYPE_CODE_PTR && t->is_const ()
		  && t->is_volatile () && t->is_restrict ());

      t = resolve (pmem);
      SELF_CHECK (t->code () == TYPE_CODE_MEMBERPTR
		  && TYPE_SELF_TYPE (t) == resolve (cls));

      for (uint32_t bad : { ptr32, pmem_short, pmem_int, pmfunc, cut })
	SELF_CHECK (resolve (bad)->code () == TYPE_CODE_ERROR);
    });
}

/* Test the types LF_ARRAY records give.  Check that the element count is
   the array's size divided by the element size, that a size of 0 leaves
   the upper bound undefined, and that the array gets the record's name.
   Check that an index type GDB's ranges cannot take, void or an empty
   structure, is replaced, and that a record cut short gives the
   unsupported type.  */

static void
test_array_types ()
{
  test_tpi tpi;
  uint32_t fwd = tpi.add (LF_STRUCTURE,
			  tag_body (LF_STRUCTURE, 0, "Fwd", CV_PROP_FWDREF));
  uint32_t arr = tpi.add (LF_ARRAY, array_body (0x74, 0x23, 12, "arr"));
  uint32_t empty = tpi.add (LF_ARRAY, array_body (0x74, 0x23, 0, ""));
  uint32_t odd = tpi.add (LF_ARRAY, array_body (0x74, 0x23, 10, ""));
  uint32_t void_idx = tpi.add (LF_ARRAY, array_body (0x74, 0x03, 8, ""));
  uint32_t fwd_idx = tpi.add (LF_ARRAY, array_body (0x74, fwd, 8, ""));
  uint32_t cut = tpi.add (LF_ARRAY, gdb::byte_vector (4, 0));

  test_read_modules ({}, {}, tpi, nullptr, [&] (pdb_per_objfile *pdb)
    {
      auto high = [] (type *t)
	{
	  return t->index_type ()->bounds ()->high;
	};

      type *t = &pdb_tpi_resolve_type (pdb, arr);
      SELF_CHECK (t->code () == TYPE_CODE_ARRAY && t->length () == 12
		  && high (t).is_constant () && high (t).const_val () == 2
		  && t->name () != nullptr && strcmp (t->name (), "arr") == 0);

      t = &pdb_tpi_resolve_type (pdb, empty);
      SELF_CHECK (t->code () == TYPE_CODE_ARRAY
		  && high (t).kind () == PROP_UNDEFINED);

      t = &pdb_tpi_resolve_type (pdb, odd);
      SELF_CHECK (t->length () == 10 && high (t).const_val () == 1);

      for (uint32_t bad_idx : { void_idx, fwd_idx })
	{
	  t = &pdb_tpi_resolve_type (pdb, bad_idx);
	  SELF_CHECK (t->code () == TYPE_CODE_ARRAY
		      && is_integral_type (t->index_type ()->target_type ())
		      && high (t).const_val () == 1);
	}

      SELF_CHECK (pdb_tpi_resolve_type (pdb, cut).code () == TYPE_CODE_ERROR);
    });
}

/* Test the types LF_MODIFIER and LF_BITFIELD records give.  Check that a
   modifier adds its qualifiers to those its type already has, and that a
   bit-field gives its underlying type.  Check that a record of either
   kind cut short gives the unsupported type.  */

static void
test_modifier_bitfield_types ()
{
  test_tpi tpi;
  auto modifier = [&] (uint32_t target, uint16_t attr)
    {
      gdb::byte_vector body;
      put_u32 (body, 0, target);
      body.push_back (attr & 0xff);
      body.push_back (attr >> 8);
      return tpi.add (LF_MODIFIER, body);
    };
  uint32_t c = modifier (0x74, CV_MODIFIER_CONST);
  uint32_t v = modifier (0x74, CV_MODIFIER_VOLATILE);
  uint32_t cv = modifier (0x74, CV_MODIFIER_CONST | CV_MODIFIER_VOLATILE);
  uint32_t none = modifier (0x74, 0);
  uint32_t vc = modifier (c, CV_MODIFIER_VOLATILE);
  gdb::byte_vector bits;
  put_u32 (bits, 0, 0x75);
  bits.push_back (3);
  bits.push_back (2);
  uint32_t bitfield = tpi.add (LF_BITFIELD, bits);
  uint32_t mod_cut = tpi.add (LF_MODIFIER, gdb::byte_vector (4, 0));
  uint32_t bit_cut = tpi.add (LF_BITFIELD, gdb::byte_vector (4, 0));

  test_read_modules ({}, {}, tpi, nullptr, [&] (pdb_per_objfile *pdb)
    {
      auto resolve = [&] (uint32_t ti)
	{
	  return &pdb_tpi_resolve_type (pdb, ti);
	};

      SELF_CHECK (resolve (c)->is_const () && !resolve (c)->is_volatile ());
      SELF_CHECK (!resolve (v)->is_const () && resolve (v)->is_volatile ());
      SELF_CHECK (resolve (cv)->is_const () && resolve (cv)->is_volatile ());
      SELF_CHECK (!resolve (none)->is_const ()
		  && resolve (none)->code () == TYPE_CODE_INT);
      SELF_CHECK (resolve (vc)->is_const () && resolve (vc)->is_volatile ());

      type *t = resolve (bitfield);
      SELF_CHECK (t->code () == TYPE_CODE_INT && t->is_unsigned ()
		  && t->length () == 4);

      SELF_CHECK (resolve (mod_cut)->code () == TYPE_CODE_ERROR);
      SELF_CHECK (resolve (bit_cut)->code () == TYPE_CODE_ERROR);
    });
}

/* Test the incomplete types of tag records.  Check that a structure, a
   class and a union get the name and size of their record, and that an
   LF_CLASS is declared a class.  Check that an enumeration gets the size
   and signedness of its underlying type, and int when that is not an
   integer.  Check that a size cut short gives the unsupported type.  */

static void
test_incomplete_tags ()
{
  test_tpi tpi;
  uint32_t s = tpi.add (LF_STRUCTURE, tag_body (LF_STRUCTURE, 8, "S"));
  uint32_t c = tpi.add (LF_CLASS, tag_body (LF_CLASS, 4, "C"));
  gdb::byte_vector big = tag_body (LF_UNION, 0, "U");
  /* An LF_USHORT size, 0x9000.  */
  big.resize (8);
  big.insert (big.end (), { 0x02, 0x80, 0x00, 0x90, 'U', 0 });
  uint32_t u = tpi.add (LF_UNION, big);
  uint32_t e = tpi.add (LF_ENUM, enum_body (0x75, "E"));
  uint32_t e_ptr = tpi.add (LF_ENUM, enum_body (0x0603, "EP"));
  gdb::byte_vector cut = tag_body (LF_STRUCTURE, 0, "");
  /* An LF_ULONG size with two of its four bytes.  */
  cut.resize (16);
  for (gdb_byte b : { 0x04, 0x80, 0x01, 0x00 })
    cut.push_back (b);
  uint32_t s_cut = tpi.add (LF_STRUCTURE, cut);

  test_read_modules ({}, {}, tpi, nullptr, [&] (pdb_per_objfile *pdb)
    {
      auto resolve = [&] (uint32_t ti)
	{
	  return &pdb_tpi_resolve_type (pdb, ti);
	};
      auto named = [] (type *t, const char *name)
	{
	  return t->name () != nullptr && strcmp (t->name (), name) == 0;
	};

      type *t = resolve (s);
      SELF_CHECK (t->code () == TYPE_CODE_STRUCT && t->length () == 8
		  && named (t, "S") && !t->is_declared_class ());
      t = resolve (c);
      SELF_CHECK (t->code () == TYPE_CODE_STRUCT && t->length () == 4
		  && named (t, "C") && t->is_declared_class ());
      t = resolve (u);
      SELF_CHECK (t->code () == TYPE_CODE_UNION && t->length () == 0x9000
		  && named (t, "U"));
      t = resolve (e);
      SELF_CHECK (t->code () == TYPE_CODE_ENUM && t->length () == 4
		  && t->is_unsigned () && named (t, "E"));
      t = resolve (e_ptr);
      SELF_CHECK (t->code () == TYPE_CODE_ENUM && t->length () == 4
		  && t->target_type ()->code () == TYPE_CODE_INT);
      SELF_CHECK (resolve (s_cut)->code () == TYPE_CODE_ERROR);
    });
}

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
  /* Check an inverted range, a range of more records than 64 bytes hold,
     a range that overflows 32 bits, and a base other than 0x1000.  */
  const range_case ranges[] = { { 0x1000, 0xfff }, { 0x1000, 0x7fffffff },
				{ 0x1000, 0xffffffff }, { 0x2000, 0x2000 } };
  for (const range_case &rc : ranges)
    SELF_CHECK (read (type_stream (TPI_VERSION_V80, 56, rc.begin, rc.end, 64))
		.find ("TPI: invalid type index range") == 0);

  SELF_CHECK (read (type_stream (20040204, 56, 0x1000, 0x1000, 0))
	      .find ("TPI unknown version") == 0);
  SELF_CHECK (read (type_stream (TPI_VERSION_V80, 55, 0x1000, 0x1000, 0))
	      .find ("header size 55 too small") != std::string::npos);

  /* Check records that run past the stream, also when the sum wraps 32
     bits.  */
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

/* Test procedure types the reader must stop on.  Build one returning
   itself, one returning a simple type of an undefined pointer mode, and a
   chain of 20000 distinct procedure types, each returning the next.  Check
   that the chain is cut at a bounded depth.  Also check that a
   procedure's type index that names no procedure type gives a function
   returning void, as GDB requires a function to have a function type.  */

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

      /* Check a type index of no type, one of a simple type and one of an
	 argument list.  */
      for (uint32_t ti : { 0u, 0x74u, none })
	{
	  type &f = pdb_tpi_resolve_function_type (pdb, ti);
	  SELF_CHECK (f.code () == TYPE_CODE_FUNC && !f.is_prototyped ()
		      && f.target_type ()->code () == TYPE_CODE_VOID);
	}
    });
}

/* Test that a procedure type belongs to the objfile even when it returns
   a builtin type, which belongs to the architecture, and that the untyped
   function type does too.  A copy Python keeps of one then survives the
   objfile's unload, as GDB preserves the objfile's types.  */

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
   take too long: more parameters than PDB_MAX_PARAMS, or a list that
   prints longer than PDB_MAX_PARAMS_LENGTH, as one of pointers to
   functions whose parameters are pointers to functions doubles at each
   level.  Such a function keeps its bare name, and its type's list is
   not tried again for the next function of that type.  */

static void
test_function_names ()
{
  test_tpi tpi;
  uint32_t one_int = tpi.procedure (0x74, tpi.arglist ({ 0x74 }), 1);
  uint32_t many = tpi.procedure (3, tpi.arglist (std::vector<uint32_t>
						  (2000, 0x74)), 2000);
  uint32_t bomb = tpi.procedure (3, tpi.arglist ({}), 0);
  for (int i = 0; i < 30; i++)
    {
      uint32_t ptr = tpi.pointer (bomb);
      bomb = tpi.procedure (3, tpi.arglist ({ ptr, ptr }), 2);
    }

  test_read_modules ({}, {}, tpi, nullptr, [&] (pdb_per_objfile *pdb)
    {
      auto name = [&] (const char *bare, uint32_t ti, enum language lang)
	{
	  return std::string (pdb_function_name
			      (pdb, bare, &pdb_tpi_resolve_type (pdb, ti),
			       lang));
	};
      SELF_CHECK (name ("f", one_int, language_cplus_) == "f(int)");
      SELF_CHECK (name ("f", one_int, language_c) == "f");

      SELF_CHECK (name ("big", many, language_cplus_) == "big");
      type *many_type = &pdb_tpi_resolve_type (pdb, many);
      auto printed = pdb->tpi.printed_param_lists.find (many_type);
      SELF_CHECK (printed != pdb->tpi.printed_param_lists.end ()
		  && printed->second == nullptr);
      SELF_CHECK (name ("big2", many, language_cplus_) == "big2");

      SELF_CHECK (name ("bomb", bomb, language_cplus_) == "bomb");
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
  selftests::register_test ("pdb-cv-read-numeric",
			    pdb::selftests::test_cv_read_numeric);
  selftests::register_test ("pdb-pointer-types",
			    pdb::selftests::test_pointer_types);
  selftests::register_test ("pdb-array-types",
			    pdb::selftests::test_array_types);
  selftests::register_test ("pdb-modifier-bitfield-types",
			    pdb::selftests::test_modifier_bitfield_types);
  selftests::register_test ("pdb-incomplete-tags",
			    pdb::selftests::test_incomplete_tags);
  selftests::register_test ("pdb-type-stream-headers",
			    pdb::selftests::test_type_stream_headers);
#endif
}
