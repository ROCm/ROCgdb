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

/* TPI (Type Program Information) and IPI (Id Program Information) stream
   reader: builds the GDB types and symbols needed for type lookup and for
   interpreting program values.

   References:
     - microsoft-pdb: https://github.com/microsoft/microsoft-pdb
     - LLVM docs: https://llvm.org/docs/PDB/TpiStream.html

   TPI (stream 2) describes types, their members and function signatures.
   IPI (stream 4) supplies function identities used to recover
   inline-function names and their TPI signatures.

   Records have consecutive indices and variable lengths.  We scan each
   stream once to build an array of record descriptors containing the kind,
   payload pointer and length.  A record can then be found directly through
   types[ti - type_idx_begin].  The record helpers only read records; the
   cooked-index worker calls them from other threads through
   pdb_ipi_lookup_inlinee.

   Records refer to other types by TI, while debugger commands request
   types by name.  We therefore also build a map from each struct, class,
   union or enum name to one TI.  A type may have both a forward record and
   a definition; this map lets us associate each forward TI with its
   definition TI before constructing GDB types.  Additional maps connect
   nested types and static members to their class, and enumerator names to
   the enums that declare them.  These support namespace reconstruction and
   lookup without first building every type.

   GDB types are constructed as needed by resolving their referenced TIs.
   A separate cache retains the resulting type objects for reuse.  Structs
   and unions are first cached as shells containing their name, kind and
   size, then their field lists are queued for population.  Recursive
   references reuse the same shell; its members are read before the
   outermost resolve call returns.  Base-class references request immediate
   population so virtual-slot recovery can use the base's method tables.
   Enums are populated at once, since enumerators reference no types.  A
   forward record whose tag the TPI never defines becomes a stub.

   Field-list processing builds members, bases, nested declarations and
   method groups, then attaches them to the owning type.  Finally, the
   symbol-building helpers publish tags, namespaces and enum constants
   for GDB's name lookup.  Lazy loading and --readnow use these same
   construction routines; they differ in when the work is requested.

   Unsupported leaves, including LF_INTERFACE, resolve to TYPE_CODE_ERROR
   named "<unsupported PDB type>".  Template instantiations can be read as
   ordinary tagged types, but template parameters are not decoded.  */

#include "symtab.h"
#include "gdbtypes.h"
#include "objfiles.h"
#include "buildsym.h"
#include "complaints.h"
#include "symfile.h"
#include "typeprint.h"
#include "cp-support.h"
#include "language.h"
#include "pdb/pdb-internal.h"
#include "gdbsupport/scope-exit.h"
#include <string.h>
#include <algorithm>
#include <chrono>

namespace pdb
{

/* Raw TPI/IPI record access and decoding.  */

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

  /* Simple leaf value - encoded directly in the leaf (2 bytes) */
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
      pdb_dbg_printf ("invalid/truncated numeric leaf 0x%04x", leaf);
      return 0;
    }
}

/* Read a NUL-terminated string from a bounded buffer.  Returns the string
   pointer if a NUL byte exists in [p, end), nullptr otherwise.  */

const char *
pdb_extract_string (const gdb_byte *p, const gdb_byte *end)
{
  if (p >= end)
    return nullptr;
  if (memchr (p, '\0', end - p) == nullptr)
    return nullptr;
  return (const char *) p;
}

/* Check that a type record has at least MIN_SIZE bytes of data.
   Returns true if OK.  Emits a warning and returns false if truncated.  */

static bool
pdb_check_record_size (const pdb_tpi_type *rec, uint32_t min_size,
		       const char *name)
{
  if (rec->data_len >= min_size)
    return true;
  pdb_warning ("%s record truncated (got %u)", name, rec->data_len);
  return false;
}

/* Return the raw TPI/IPI record at TYPE_IDX, or nullptr if the record array
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

/* See pdb-internal.h.  */

uint16_t
pdb_tpi_tag_props (const pdb_tpi_context *tpi, uint32_t type_idx)
{
  const pdb_tpi_type *rec = pdb_tpi_get_type (tpi, type_idx);
  if (rec == nullptr)
    return 0;

  switch (rec->leaf)
    {
    case LF_CLASS:
    case LF_STRUCTURE:
      if (rec->data_len < LF_STRUCT_PROPERTY_OFFS + 2)
	return 0;
      return read_u16 (rec->data + LF_STRUCT_PROPERTY_OFFS);

    case LF_UNION:
      if (rec->data_len < LF_UNION_PROPERTY_OFFS + 2)
	return 0;
      return read_u16 (rec->data + LF_UNION_PROPERTY_OFFS);

    case LF_ENUM:
      if (rec->data_len < LF_ENUM_PROPERTY_OFFS + 2)
	return 0;
      return read_u16 (rec->data + LF_ENUM_PROPERTY_OFFS);

    default:
      return 0;
    }
}

/* See pdb-internal.h.  */

bool
pdb_tpi_type_is_fwdref (const pdb_tpi_context *tpi, uint32_t type_idx)
{
  return (pdb_tpi_tag_props (tpi, type_idx) & CV_PROP_FWDREF) != 0;
}

/* Return the tag name stored in a tagged-type record
   (LF_CLASS/STRUCTURE/UNION/ENUM).  For MSVC this is the fully
   qualified name (e.g. "ns::Foo").  */

static const char *
pdb_tpi_extract_tag_name (const pdb_tpi_type *rec)
{
  switch (rec->leaf)
    {
    case LF_CLASS:
    case LF_STRUCTURE:
      if (rec->data_len > LF_STRUCT_MIN_SIZE)
	{
	  uint64_t size_val;
	  uint32_t nr = pdb_cv_read_numeric (rec->data + LF_STRUCT_DATA_OFFS,
					     rec->data_len
					       - LF_STRUCT_DATA_OFFS,
					     &size_val);
	  if (nr > 0)
	    return pdb_extract_string (rec->data + LF_STRUCT_DATA_OFFS + nr,
				       rec->data + rec->data_len);
	}
      return nullptr;

    case LF_UNION:
      if (rec->data_len > LF_UNION_MIN_SIZE)
	{
	  uint64_t size_val;
	  uint32_t nr = pdb_cv_read_numeric (rec->data + LF_UNION_DATA_OFFS,
					     rec->data_len
					       - LF_UNION_DATA_OFFS,
					     &size_val);
	  if (nr > 0)
	    return pdb_extract_string (rec->data + LF_UNION_DATA_OFFS + nr,
				       rec->data + rec->data_len);
	}
      return nullptr;

    case LF_ENUM:
      if (rec->data_len > LF_ENUM_MIN_SIZE)
	return pdb_extract_string (rec->data + LF_ENUM_NAME_OFFS,
				   rec->data + rec->data_len);
      return nullptr;

    default:
      return nullptr;
    }
}

/* See pdb-internal.h.  */

const char *
pdb_tpi_tag_name (const pdb_tpi_context *tpi, uint32_t type_idx)
{
  const pdb_tpi_type *rec = pdb_tpi_get_type (tpi, type_idx);
  if (rec == nullptr)
    return nullptr;

  return pdb_tpi_extract_tag_name (rec);
}

/* Return the decorated unique name of the tagged record at TYPE_IDX, or
   nullptr when the record has none (CV_PROP_HASUNIQUENAME clear).  It
   follows the display name and, unlike it, differs between same-named tags
   in different scopes, such as two anonymous namespaces.  */

static const char *
pdb_tpi_unique_name (const pdb_tpi_context *tpi, uint32_t type_idx)
{
  if ((pdb_tpi_tag_props (tpi, type_idx) & CV_PROP_HASUNIQUENAME) == 0)
    return nullptr;

  const pdb_tpi_type *rec = pdb_tpi_get_type (tpi, type_idx);
  const char *name = pdb_tpi_extract_tag_name (rec);
  if (name == nullptr)
    return nullptr;

  return pdb_extract_string ((const gdb_byte *) name + strlen (name) + 1,
			     rec->data + rec->data_len);
}

/* See pdb-internal.h.  */

uint32_t
pdb_tpi_mfunction_class (const pdb_tpi_context *tpi, uint32_t type_idx)
{
  const pdb_tpi_type *rec = pdb_tpi_get_type (tpi, type_idx);
  if (rec == nullptr || rec->leaf != LF_MFUNCTION
      || rec->data_len < LF_MFUNC_SIZE)
    return 0;

  return read_u32 (rec->data + LF_MFUNC_CLASSTYPE_OFFS);
}

/* Return the field-list type index a tagged record points at, or 0.  */

static uint32_t
pdb_tag_fieldlist_ti (const pdb_tpi_type *rec)
{
  switch (rec->leaf)
    {
    case LF_CLASS:
    case LF_STRUCTURE:
      if (rec->data_len < LF_STRUCT_FIELDLIST_OFFS + 4)
	return 0;
      return read_u32 (rec->data + LF_STRUCT_FIELDLIST_OFFS);

    case LF_UNION:
      if (rec->data_len < LF_UNION_FIELDLIST_OFFS + 4)
	return 0;
      return read_u32 (rec->data + LF_UNION_FIELDLIST_OFFS);

    case LF_ENUM:
      if (rec->data_len < LF_ENUM_FIELDLIST_OFFS + 4)
	return 0;
      return read_u32 (rec->data + LF_ENUM_FIELDLIST_OFFS);

    default:
      return 0;
    }
}

/* See pdb-internal.h.  */

int
pdb_tpi_get_func_param_count (pdb_per_objfile *pdb, uint32_t type_idx)
{
  const pdb_tpi_type *rec = pdb_tpi_get_type (&pdb->tpi, type_idx);
  if (rec == nullptr)
    return 0;

  if (rec->leaf == LF_MFUNCTION && rec->data_len >= LF_MFUNC_SIZE)
    {
      /* A static method has no 'this', and its arglist has no slot for
	 one either.  */
      bool has_this = read_u32 (rec->data + LF_MFUNC_THISTYPE_OFFS) != 0;
      return read_u16 (rec->data + LF_MFUNC_PARMCOUNT_OFFS) + (has_this ? 1 : 0);
    }

  else if (rec->leaf == LF_PROCEDURE && rec->data_len >= LF_PROC_SIZE)
    return read_u16 (rec->data + LF_PROC_PARMCOUNT_OFFS);

  return 0;
}

/* CV_methodprop_e — a 3-bit field (bits 2-4) within the CV_fldattr_t
   attribute word that every LF_ONEMETHOD and LF_METHODLIST entry carries.
   It classifies the method:

     VANILLA   — ordinary non-virtual member function.
     VIRTUAL   — virtual override (not the first declaration).
     STATIC    — static member function.
     FRIEND    — friend function (not a member).
     INTRO     — first (introducing) declaration of a virtual method.
     PUREVIRT  — pure virtual override (= 0), not the first decl.
     PUREINTRO — pure virtual and the introducing declaration.

   INTRO and PUREINTRO entries have an extra uint32_t vbaseoff field
   (the vtable offset) immediately after the fixed sub-record fields.
   The parser reads each entry's attr, extracts mprop with
   CV_MPROP(), and tests CV_MPROP_HAS_VBASEOFF() to decide whether
   to skip 4 extra bytes before the method name (LF_ONEMETHOD) or
   before the next entry (LF_METHODLIST).  */
inline constexpr auto CV_MPROP_VANILLA = 0;
inline constexpr auto CV_MPROP_VIRTUAL = 1;
inline constexpr auto CV_MPROP_STATIC = 2;
inline constexpr auto CV_MPROP_FRIEND = 3;
/* Introducing virtual.  */
inline constexpr auto CV_MPROP_INTRO = 4;
inline constexpr auto CV_MPROP_PUREVIRT = 5;
/* Pure introducing virtual.  */
inline constexpr auto CV_MPROP_PUREINTRO = 6;

/* Extract CV_methodprop_e from CV_fldattr_t.  */
#define CV_MPROP(attr) (((attr) >> 2) & 0x07)

/* True when mprop is INTRO or PUREINTRO (extra uint32_t vbaseoff follows).  */
#define CV_MPROP_HAS_VBASEOFF(mprop) \
  ((mprop) == CV_MPROP_INTRO || (mprop) == CV_MPROP_PUREINTRO)

/* Advance past a single sub-record at P, returning pointer past it.
  Return nullptr for truncation or an unrecognized layout.

  Common variable-length patterns are:
     1. fixed fields + numeric leaf + NUL name  (LF_MEMBER, LF_ENUMERATE)
     2. fixed fields + numeric leaf (no name)   (LF_BCLASS)
    3. fixed fields + NUL name (no numeric)
       (LF_STMEMBER, LF_NESTTYPE, LF_METHOD)
     Virtual bases carry two numeric leaves; LF_ONEMETHOD may carry a vbaseoff
     before its name.  Fixed-size records need only their size check.  */

static const gdb_byte *
pdb_fieldlist_skip_record (const gdb_byte *p, const gdb_byte *end,
			   uint16_t leaf)
{
  using bytep = const gdb_byte *;

  /* Skip past numeric leaf + NUL-terminated name starting at DATA_OFFS.  */
  auto skip_numeric_name = [&] (uint32_t data_offs) -> bytep
    {
      if (p + data_offs > end)
	return nullptr;
      const gdb_byte *d = p + data_offs;
      uint64_t dummy;
      uint32_t nr = pdb_cv_read_numeric (d, (uint32_t) (end - d), &dummy);
      if (nr == 0)
	return nullptr;

      d += nr;
      const auto nul = (const gdb_byte *) memchr (d, 0, (size_t) (end - d));
      /* Advance the pointer past the null-terminated name.  */
      return nul ? nul + 1 : nullptr;
    };

  /* Skip past numeric leaf (no trailing name) starting at DATA_OFFS.  */
  auto skip_numeric = [&] (uint32_t data_offs) -> bytep
    {
      if (p + data_offs > end)
	return nullptr;

      const gdb_byte *d = p + data_offs;
      uint64_t dummy;
      uint32_t nr = pdb_cv_read_numeric (d, (uint32_t) (end - d), &dummy);
      return (nr == 0) ? nullptr : d + nr;
    };

  /* Skip past NUL-terminated name starting at NAME_OFFS.  */
  auto skip_name = [&] (uint32_t name_offs) -> bytep
    {
      if (p + name_offs > end)
	return nullptr;

      const gdb_byte *d = p + name_offs;
      /* Advance the pointer past the null-terminated name.  */
      const auto nul = (const gdb_byte *) memchr (d, 0, end - d);
      return nul ? nul + 1 : nullptr;
    };

  switch (leaf)
    {
    case LF_MEMBER:
      return skip_numeric_name (LF_MEMBER_DATA_OFFS);

    case LF_ENUMERATE:
      return skip_numeric_name (LF_ENUMERATE_DATA_OFFS);

    case LF_BCLASS:
      return skip_numeric (LF_BCLASS_DATA_OFFS);

    case LF_BINTERFACE:
      return skip_numeric (LF_BCLASS_DATA_OFFS);

    case LF_VBCLASS:
    case LF_IVBCLASS:
      {
	/* Two variable-length numeric leaves.  */
	if (p + LF_VBCLASS_DATA_OFFS > end)
	  return nullptr;

	const gdb_byte *d = p + LF_VBCLASS_DATA_OFFS;
	uint64_t dummy;
	auto nr = pdb_cv_read_numeric (d, (uint32_t) (end - d), &dummy);
	if (nr == 0)
	  return nullptr;
	d += nr;
	nr = pdb_cv_read_numeric (d, (uint32_t) (end - d), &dummy);
	if (nr == 0)
	  return nullptr;

	return d + nr;
      }

    case LF_STMEMBER:
      return skip_name (LF_STMEMBER_NAME_OFFS);

    case LF_NESTTYPE:
      return skip_name (LF_NESTTYPE_NAME_OFFS);

    case LF_NESTTYPEEX:
      return skip_name (LF_NESTTYPE_NAME_OFFS);

    case LF_VFUNCOFF:
      return (p + LF_VFUNCOFF_SIZE <= end) ? p + LF_VFUNCOFF_SIZE : nullptr;

    case LF_METHOD:
      return skip_name (LF_METHOD_NAME_OFFS);

    case LF_VFUNCTAB:
      return (p + LF_VFUNCTAB_SIZE <= end) ? p + LF_VFUNCTAB_SIZE : nullptr;

    case LF_ONEMETHOD:
      {
	if (p + LF_ONEMETHOD_DATA_OFFS > end)
	  return nullptr;

	uint16_t mattr = read_u16 (p + LF_ONEMETHOD_ATTR_OFFS);
	const gdb_byte *d = p + LF_ONEMETHOD_DATA_OFFS;
	if (CV_MPROP_HAS_VBASEOFF (CV_MPROP (mattr)))
	  {
	    /* Skip vbaseoff.  */
	    d += 4;
	  }
	if (d > end)
	  return nullptr;

	const auto nul = (const gdb_byte *) memchr (d, 0, end - d);
	return nul ? nul + 1 : nullptr;
      }

    case LF_INDEX:
      return (p + LF_INDEX_SIZE <= end) ? p + LF_INDEX_SIZE : nullptr;

    default:
      return nullptr;
    }
}

/* Parsed fields common to all compound type records.  */
struct pdb_compound_fields
{
  /* Number of members.  */
  uint16_t count;
  /* CV_prop_t flags.  */
  uint16_t property;
  /* Type index of LF_FIELDLIST.  */
  uint32_t fieldlist_ti;
  /* Struct/union size, or underlying type length.  */
  uint64_t byte_size;
  const char *name;
};

/* Parse common fields from LF_STRUCTURE/CLASS/UNION records.  Field layout is
   defined by LF_STRUCT_* / LF_UNION_* offsets in pdb-internal.h.  Returns
   std::nullopt on truncated/malformed records.  */

static std::optional<pdb_compound_fields>
pdb_parse_tagged_record (const pdb_tpi_type *rec, uint32_t min_size,
			 uint32_t fieldlist_offs, uint32_t data_offs,
			 const char *kind_name)
{
  if (!pdb_check_record_size (rec, min_size, kind_name))
    return std::nullopt;

  /* data_offs must be within the record so the numeric leaf has something
     to consume.  */
  if (data_offs > rec->data_len)
    return std::nullopt;

  pdb_compound_fields out;
  out.count = read_u16 (rec->data + LF_STRUCT_COUNT_OFFS);
  out.property = read_u16 (rec->data + LF_STRUCT_PROPERTY_OFFS);
  out.fieldlist_ti = read_u32 (rec->data + fieldlist_offs);

  /* The struct/union byte size is variable-length encoded (numeric leaf).
     The buffer that holds the leaf is everything from data_offs to the
     end of the record.  */
  gdb::array_view<const gdb_byte> size_buf (rec->data + data_offs,
					    rec->data_len - data_offs);

  uint32_t nr = pdb_cv_read_numeric (size_buf.data (), size_buf.size (),
				     &out.byte_size);
  if (nr == 0)
    return std::nullopt;

  /* The trailing name must fit and be NUL-terminated within the record.  */
  const gdb_byte *name_p = size_buf.data () + nr;
  const gdb_byte *name_end = size_buf.data () + size_buf.size ();
  const char *name = pdb_extract_string (name_p, name_end);
  if (name == nullptr)
    return std::nullopt;

  out.name = name;
  return out;
}

/* TPI and IPI stream loading and load-time indexes.  */

/* TPI and IPI streams.
   https://llvm.org/docs/PDB/TpiStream.html

   The TPI stream (stream 2) and IPI stream (stream 4) contain type records that
   describe all types used by the program.  Symbols reference types through a
   32-bit Type Index (TI).  Type indices < TypeIndexBegin (0x1000) are built-in
   types whose meaning is encoded directly in the index value:
     bits  0-7  : type kind (void, int, float, …)
     bits  8-11 : type mode (direct, near ptr, far ptr, …)
   Any index >= TypeIndexBegin refers to a record in the TPI (or IPI) type
   record array.  Records form a topologically sorted DAG: record B may only
   reference record A if A's type index < B's type index.
   TPI/IPI stream layout:
     TpiStreamHeader  (56 bytes)
     Array of Type Records:
       Type Record Layout:
	  RecordLen (2 bytes)  — length of RecordKind + variable data
	  RecordKind (2 bytes) — Leaf type (LF_*)
	  RecordData (RecordLen - 2 bytes) — fields depend on RecordKind.  */

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

/* See pdb-internal.h.  */

void
pdb_parse_tpi_records (pdb_per_objfile *pdb, uint32_t stream_idx,
		       uint32_t rec_start_off, uint32_t rec_bytes,
		       const char *stream_name, pdb_tpi_context &tpi)
{
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
	pdb_error ("%s: unexpected end of records at type 0x%x", stream_name,
		   tpi.type_idx_begin + idx);

      /* RecordLen counts every byte after itself (RecordKind + data);
	 read it first to learn the record's total size.  */
      const gdb_byte *h = pdb_stream_bytes (pdb, stream_idx, off,
					    CV_REC_HDR_SIZE, scratch);
      uint16_t rec_len = read_u16 (h);
      uint16_t rec_kind = read_u16 (h + 2);

      if (rec_len < 2)
	pdb_error ("%s: type 0x%x has invalid RecordLen %u", stream_name,
		   tpi.type_idx_begin + idx, rec_len);

      /* Full record = RecordLen field (2 bytes) + rec_len bytes after it.  */
      uint32_t rec_size = rec_len + 2;
      if (off + rec_size > rec_area_end)
	pdb_error ("%s: type 0x%x overflows record area", stream_name,
		   tpi.type_idx_begin + idx);

      /* Data is what follows RecordKind, so drop its 2 bytes from rec_len.  */
      uint32_t data_len = rec_len - 2;
      const gdb_byte *data
	= pdb_stream_bytes (pdb, stream_idx, off + CV_REC_HDR_SIZE, data_len,
			    scratch);

      /* Scratch-backed payloads must be copied before the buffer is reused.
    	 Mapping-backed payloads already have the objfile's lifetime.  */
      if (data == scratch.data ())
	data = (const gdb_byte *) obstack_copy (
		 &pdb->objfile->objfile_obstack, data, data_len);

      types[idx].leaf = rec_kind;
      types[idx].length = rec_len;
      types[idx].data = data;
      types[idx].data_len = data_len;

      idx++;
      off += rec_size;
    }

  tpi.types = types;

  double ms = std::chrono::duration<double, std::milli> (
		clock::now () - start).count ();
  pdb_dbg_printf ("%s: indexed %u type records in %.2f ms", stream_name,
		  num_types, ms);
}

/* See pdb-internal.h.  */

const char *
pdb_canonical_name (pdb_per_objfile *pdb, const char *name)
{
  if (name == nullptr)
    return nullptr;

  /* A name that already carries a parameter list was printed by
     c_type_print_args and is canonical; canonicalizing it again would
     rewrite "(void)" as "()".  */
  if (strchr (name, '(') != nullptr)
    return name;

  gdb::unique_xmalloc_ptr<char> canon = cp_canonicalize_string (name);
  if (canon == nullptr)
    return name;

  return obstack_strdup (&pdb->objfile->objfile_obstack, canon.get ());
}

/* Whether CHILD is spelled "OWNER::LEAF".  */

static bool
pdb_name_is_member_of (const char *child, const char *owner, const char *leaf)
{
  size_t olen = strlen (owner);

  return (strncmp (child, owner, olen) == 0
	  && child[olen] == ':' && child[olen + 1] == ':'
	  && strcmp (child + olen + 2, leaf) == 0);
}

/* The tag selected as owner of each field list, and the field lists that
   more than one tag refers to directly.  */

struct pdb_fieldlist_owners
{
  std::unordered_map<uint32_t, uint32_t> owner;
  std::unordered_set<uint32_t> shared;
};

/* Give each field list the first tag that refers to it in TPI order, since
   a field list does not name its owner.  Carry that owner through LF_INDEX
   continuations until a continuation already has one.  */

static pdb_fieldlist_owners
pdb_assign_fieldlist_owners (const pdb_tpi_context &tpi,
			     pdb_nesting_stats &stats)
{
  pdb_fieldlist_owners result;
  uint32_t count = tpi.type_idx_end - tpi.type_idx_begin;
  result.owner.reserve (count);

  for (uint32_t i = 0; i < count; i++)
    {
      uint32_t fl = pdb_tag_fieldlist_ti (&tpi.types[i]);
      if (fl == 0)
	continue;

      stats.tags_scanned++;
      if (!result.owner.emplace (fl, tpi.type_idx_begin + i).second)
	{
	  stats.shared_fieldlists++;
	  result.shared.insert (fl);
	}
    }

  /* The LF_INDEX continuation named by field list REC, or 0.  */
  auto continuation = [] (const pdb_tpi_type *rec) -> uint32_t
    {
      const gdb_byte *p = rec->data;
      const gdb_byte *end = rec->data + rec->data_len;
      uint32_t next = 0;

      while (p < end)
	{
	  p = align_up (p, 4);
	  if (p + 2 > end)
	    break;

	  uint16_t leaf = read_u16 (p);
	  if (leaf == LF_INDEX && p + LF_INDEX_TYPE_OFFS + 4 <= end)
	    next = read_u32 (p + LF_INDEX_TYPE_OFFS);

	  p = pdb_fieldlist_skip_record (p, end, leaf);
	  if (p == nullptr)
	    break;
	}

      return next;
    };

  std::vector<std::pair<uint32_t, uint32_t>> seeds (result.owner.begin (),
						    result.owner.end ());
  for (const auto &[fl_ti, owner] : seeds)
    {
      const pdb_tpi_type *cont = pdb_tpi_get_type (&tpi, fl_ti);
      while (cont != nullptr && cont->leaf == LF_FIELDLIST)
	{
	  uint32_t next = continuation (cont);
	  if (next == 0 || !result.owner.emplace (next, owner).second)
	    break;
	  cont = pdb_tpi_get_type (&tpi, next);
	}
    }

  return result;
}

/* Handle the LF_NESTTYPE at P in the field list of the tag OWNER_TI, whose
   name is OWNER_TAG.  If the nested type is a tag named exactly
   "OWNER_TAG::name", set CHILD_TO_OWNER[that tag's TI] = OWNER_TI unless
   it already has an owner.  Otherwise the LF_NESTTYPE is a member typedef
   or alias for a type declared elsewhere, and is only counted.  SHARED is
   true when other tags also refer to this field list.  */

static void
pdb_record_nested_parent (const pdb_tpi_context &tpi, const gdb_byte *p,
			  const gdb_byte *end, uint32_t owner_ti,
			  const char *owner_tag, bool shared,
			  std::unordered_map<uint32_t, uint32_t>
			    &child_to_owner,
			  pdb_nesting_stats &stats)
{
  uint32_t child = read_u32 (p + LF_NESTTYPE_TYPE_OFFS);
  const char *nested = pdb_extract_string (p + LF_NESTTYPE_NAME_OFFS, end);
  if (child == 0 || nested == nullptr)
    return;

  const char *child_tag = pdb_tpi_tag_name (&tpi, child);
  if (child_tag == nullptr
      || !pdb_name_is_member_of (child_tag, owner_tag, nested))
    {
      stats.alias_edges++;
      return;
    }

  if (child_to_owner.emplace (child, owner_ti).second)
    {
      stats.edges++;
      if (shared)
	stats.ambiguous_edges++;
    }
}

/* Walk the field list REC of the tag OWNER_TI, whose name is OWNER_TAG,
   and record that tag as the declaring class of what REC declares:

     - each nested tag (LF_NESTTYPE): CHILD_TO_OWNER[nested TI] = OWNER_TI,
       see pdb_record_nested_parent;
     - each static data member (LF_STMEMBER "m"):
       tpi.static_member_owner["OWNER_TAG::m"] = OWNER_TI.

   An existing entry is kept.  OWNER_TI is the tag that
   pdb_assign_fieldlist_owners chose for REC.  SHARED is true when other
   tags also refer to REC; STATS counts nested entries recorded from such a
   list as ambiguous.  */

static void
pdb_record_member_parents (pdb_tpi_context &tpi, const pdb_tpi_type *rec,
			   uint32_t owner_ti, const char *owner_tag,
			   bool shared,
			   std::unordered_map<uint32_t, uint32_t>
			     &child_to_owner,
			   pdb_nesting_stats &stats)
{
  const gdb_byte *p = rec->data;
  const gdb_byte *end = rec->data + rec->data_len;

  while (p < end)
    {
      p = align_up (p, 4);
      if (p + 2 > end)
	break;

      uint16_t leaf = read_u16 (p);
      stats.subrecords_walked++;

      if (leaf == LF_NESTTYPE && p + LF_NESTTYPE_TYPE_OFFS + 4 <= end)
	pdb_record_nested_parent (tpi, p, end, owner_ti, owner_tag, shared,
				  child_to_owner, stats);
      else if (leaf == LF_STMEMBER)
	{
	  /* The member is declared bare here; the symbol that gives it an
	     address spells it out under the declaring tag.  */
	  const char *member = pdb_extract_string (p + LF_STMEMBER_NAME_OFFS,
						   end);
	  if (member != nullptr
	      && tpi.static_member_owner.emplace (std::string (owner_tag)
						  + "::" + member,
						  owner_ti).second)
	    stats.static_members++;
	}

      p = pdb_fieldlist_skip_record (p, end, leaf);
      if (p == nullptr)
	break;
    }
}

/* See pdb-internal.h.  */

void
pdb_build_parent_map (pdb_per_objfile *pdb,
		      std::unordered_map<uint32_t, uint32_t> &child_to_owner,
		      pdb_nesting_stats &stats)
{
  pdb_tpi_context &tpi = pdb->tpi;
  if (tpi.types == nullptr || tpi.type_idx_end <= tpi.type_idx_begin)
    return;

  using clock = std::chrono::steady_clock;
  auto start = clock::now ();

  pdb_fieldlist_owners owners = pdb_assign_fieldlist_owners (tpi, stats);

  uint32_t count = tpi.type_idx_end - tpi.type_idx_begin;
  for (uint32_t i = 0; i < count; i++)
    {
      const pdb_tpi_type *rec = &tpi.types[i];
      if (rec->leaf != LF_FIELDLIST)
	continue;

      stats.fieldlists_walked++;

      uint32_t fl_ti = tpi.type_idx_begin + i;
      auto owner = owners.owner.find (fl_ti);
      if (owner == owners.owner.end ())
	{
	  stats.unowned_continuations++;
	  continue;
	}

      const char *owner_tag = pdb_tpi_tag_name (&tpi, owner->second);
      if (owner_tag == nullptr || owner_tag[0] == '\0')
	continue;

      pdb_record_member_parents (tpi, rec, owner->second, owner_tag,
				 owners.shared.count (fl_ti) != 0,
				 child_to_owner, stats);
    }

  stats.build_ms = std::chrono::duration<double, std::milli> (
		     clock::now () - start).count ();
}

/* Add each enumerator of the LF_ENUM record REC to enumerator_names,
   mapped to TAG, the enum's canonical name.  Walks the field list like
   pdb_tpi_parse_fieldlist but reads only the names.  */

static void
pdb_index_enumerators (pdb_per_objfile *pdb, const pdb_tpi_type *rec,
		       const char *tag)
{
  uint32_t fl_ti = pdb_tag_fieldlist_ti (rec);
  if (fl_ti == 0)
    return;

  const pdb_tpi_type *fl = pdb_tpi_get_type (&pdb->tpi, fl_ti);
  if (fl == nullptr || fl->leaf != LF_FIELDLIST)
    return;

  const gdb_byte *p = fl->data;
  const gdb_byte *end = fl->data + fl->data_len;
  std::vector<uint32_t> visited { fl_ti };

  while (p < end)
    {
      p = align_up (p, 4);
      if (p + 2 > end)
	break;

      uint16_t leaf = read_u16 (p);
      const gdb_byte *next = pdb_fieldlist_skip_record (p, end, leaf);
      if (next == nullptr)
	return;

      if (leaf == LF_INDEX)
	{
	  uint32_t cont_ti = read_u32 (p + LF_INDEX_TYPE_OFFS);
	  const pdb_tpi_type *cont = pdb_tpi_get_type (&pdb->tpi, cont_ti);
	  if (cont == nullptr || cont->leaf != LF_FIELDLIST
	      || std::find (visited.begin (), visited.end (), cont_ti)
		   != visited.end ())
	    return;
	  visited.push_back (cont_ti);
	  p = cont->data;
	  end = cont->data + cont->data_len;
	  continue;
	}

      if (leaf == LF_ENUMERATE)
	{
	  const gdb_byte *d = p + LF_ENUMERATE_DATA_OFFS;
	  uint64_t value;
	  uint32_t nr = pdb_cv_read_numeric (d, (uint32_t) (next - d), &value);
	  const char *name = pdb_extract_string (d + nr, next);
	  if (nr != 0 && name != nullptr)
	    pdb->tpi.enumerator_names.emplace (std::string_view (name), tag);
	}

      p = next;
    }
}

/* See pdb-internal.h.  */

void
pdb_tpi_build_tagged_name_cache (pdb_per_objfile *pdb)
{
  pdb_tpi_context &tpi = pdb->tpi;
  if (tpi.types == nullptr || tpi.type_idx_end <= tpi.type_idx_begin)
    return;

  using clock = std::chrono::steady_clock;
  auto start = clock::now ();

  uint32_t count = tpi.type_idx_end - tpi.type_idx_begin;
  tpi.tagged_type_names.reserve (count);
  std::vector<std::pair<uint32_t, std::string_view>> fwdrefs;
  std::unordered_map<std::string_view, uint32_t> unique_definition;
  std::unordered_set<std::string_view> ambiguous_names;

  for (uint32_t i = 0; i < count; i++)
    {
      const char *name = pdb_tpi_extract_tag_name (&tpi.types[i]);
      if (name == nullptr || name[0] == '\0')
	continue;

      name = pdb_canonical_name (pdb, name);

      uint32_t ti = tpi.type_idx_begin + i;
      bool fwdref = pdb_tpi_type_is_fwdref (&tpi, ti);
      if (fwdref)
	fwdrefs.emplace_back (ti, std::string_view (name));
      else if (const char *unique = pdb_tpi_unique_name (&tpi, ti))
	unique_definition[unique] = ti;

      auto res = tpi.tagged_type_names.emplace (std::string_view (name), ti);
      if (res.second || fwdref)
	continue;

      /* A definition replaces an earlier entry; a forward record never
	 does, so the last definition wins.  */
      if (!pdb_tpi_type_is_fwdref (&tpi, res.first->second))
	ambiguous_names.insert (res.first->first);
      res.first->second = ti;
    }

  /* struct and class records are interchangeable; union and enum match
     only themselves.  */
  auto tag_kind = [&] (uint32_t ti)
    {
      uint16_t leaf = pdb_tpi_get_type (&tpi, ti)->leaf;
      return leaf == LF_CLASS ? LF_STRUCTURE : leaf;
    };

  /* Map each forward record to its definition, so pdb_tpi_resolve_type
     builds one type for both.  A forward record with a unique name takes
     the definition with the same unique name.  One without takes the
     definition of its display name, unless several definitions share that
     name.  Every other record, and a forward record left without a
     definition, keeps 0.  */
  tpi.fwdref_definition.assign (count, 0);
  for (const auto &[ti, name] : fwdrefs)
    {
      uint32_t def = 0;
      if (const char *unique = pdb_tpi_unique_name (&tpi, ti))
	{
	  auto it = unique_definition.find (unique);
	  if (it != unique_definition.end ())
	    def = it->second;
	}
      else if (ambiguous_names.count (name) == 0)
	{
	  auto it = tpi.tagged_type_names.find (name);
	  if (it != tpi.tagged_type_names.end ()
	      && !pdb_tpi_type_is_fwdref (&tpi, it->second))
	    def = it->second;
	}

      if (def != 0 && tag_kind (def) == tag_kind (ti))
	tpi.fwdref_definition[ti - tpi.type_idx_begin] = def;
    }

  /* Only the selected record of each enum is ever built, so only its
     enumerators can become symbols.  */
  for (const auto &[name, ti] : tpi.tagged_type_names)
    {
      const pdb_tpi_type *rec = pdb_tpi_get_type (&tpi, ti);
      if (rec->leaf == LF_ENUM && !pdb_tpi_type_is_fwdref (&tpi, ti))
	pdb_index_enumerators (pdb, rec, name.data ());
    }

  if (pdb_read_debug >= 1)
    {
      double ms = std::chrono::duration<double, std::milli> (
			clock::now () - start).count ();
      debug_printf ("[pdb type-index] tagged-name cache: %zu names scanned"
		    " from %u TPI records in %.2f ms\n",
		    tpi.tagged_type_names.size (), count, ms);
    }
}

/* See pdb-internal.h.  */

bool
pdb_tpi_is_tagged_type_name (const pdb_tpi_context *tpi, const char *qname)
{
  if (tpi == nullptr || qname == nullptr)
    return false;
  return tpi->tagged_type_names.find (std::string_view (qname))
	 != tpi->tagged_type_names.end ();
}

/* Index TPI (stream 2), allocate the type cache, and build the name and
  ownership maps.  Missing streams and detected format errors abort loading.  */

void
pdb_read_tpi_stream (pdb_per_objfile *pdb)
{
  auto &tpi = pdb->tpi;

  if (pdb->streams.size () <= PDB_STREAM_TPI
      || pdb->streams[PDB_STREAM_TPI].size == 0)
    {
      pdb_error ("TPI stream (index 2) missing or empty");
    }

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

  pdb_parse_tpi_records (pdb, PDB_STREAM_TPI, hdr_size, rec_bytes, "TPI", tpi);

  /* Cover simple indices [0, 0x1000) and record indices below
     type_idx_end.  */
  uint32_t cache_size = tpi.type_idx_end > 0x1000 ? tpi.type_idx_end : 0x1000;
  pdb->tpi.type_cache = OBSTACK_CALLOC (&pdb->objfile->objfile_obstack,
					cache_size, type *);

  /* Build the name -> TI cache for later use.  */
  pdb_tpi_build_tagged_name_cache (pdb);

  /* Namespace inference needs the class that owns each nested type and
     static data member.  */
  pdb_build_parent_map (pdb, pdb->tpi.nested_owner, pdb->tpi.nesting_stats);

  if (pdb_read_debug >= 1)
    debug_printf ("[pdb type-index] nested-type index: %zu edges from %zu"
		  " field lists in %.2f ms\n",
		  pdb->tpi.nesting_stats.edges,
		  pdb->tpi.nesting_stats.fieldlists_walked,
		  pdb->tpi.nesting_stats.build_ms);
}

/* See pdb-internal.h.  */

bool
pdb_read_ipi_stream (pdb_per_objfile *pdb)
{
  auto &ipi = pdb->ipi;

  if (pdb->streams.size () <= PDB_STREAM_IPI
      || pdb->streams[PDB_STREAM_IPI].size == 0)
    return false;

  size_t ipi_size = pdb->streams[PDB_STREAM_IPI].size;
  if (ipi_size < TPI_HDR_SIZE)
    {
      pdb_warning ("IPI stream too small (%zu bytes)", ipi_size);
      return false;
    }
  gdb::byte_vector ipi_hdr_scratch;
  const gdb_byte *ipi_p = pdb_stream_bytes (pdb, PDB_STREAM_IPI, 0,
					    TPI_HDR_SIZE, ipi_hdr_scratch);

  uint32_t version = read_u32 (ipi_p + TPI_HDR_VERSION_OFFS);
  uint32_t hdr_size = read_u32 (ipi_p + TPI_HDR_HEADER_SIZE_OFFS);
  uint32_t rec_bytes = read_u32 (ipi_p + TPI_HDR_TYPE_REC_BYTES_OFFS);
  ipi.type_idx_begin = read_u32 (ipi_p + TPI_HDR_TYPE_INDEX_BEGIN_OFFS);
  ipi.type_idx_end = read_u32 (ipi_p + TPI_HDR_TYPE_INDEX_END_OFFS);

  if (version != TPI_VERSION_V80)
    {
      pdb_warning ("IPI unknown version 0x%08x", version);
      return false;
    }

  if (hdr_size < TPI_HDR_SIZE || (uint64_t) hdr_size + rec_bytes > ipi_size)
    {
      pdb_warning ("IPI header/records out of range");
      return false;
    }

  uint32_t num_ids = ipi.type_idx_end - ipi.type_idx_begin;
  pdb_dbg_printf ("IPI: ID=[0x%x..0x%x)  num_ids=%u  rec_bytes=%u",
		  ipi.type_idx_begin, ipi.type_idx_end, num_ids, rec_bytes);

  /* Record-parse errors are nonfatal.  With ipi.types null, inlinee lookup
     fails and the symbol parser retains scope balance without creating an
     inline function or its line entries.  */
  try
    {
      pdb_parse_tpi_records (pdb, PDB_STREAM_IPI, hdr_size, rec_bytes,
			     "IPI", ipi);
      return true;
    }
  catch (const gdb_exception_error &e)
    {
      ipi.types = nullptr;
      pdb_warning ("IPI parse failed (%s); inline names unavailable",
		   e.what ());
      return false;
    }
}

/* IPI function identities.  */

/* Append an IPI scope string, expanding its optional substring list.  */

static bool
pdb_ipi_append_string (const pdb_tpi_context &ipi, uint32_t item_id,
		       std::string &name, std::vector<uint32_t> &active,
		       uint32_t &remaining)
{
  if (remaining == 0 || active.size () >= 64
      || std::find (active.begin (), active.end (), item_id) != active.end ())
    return false;
  --remaining;

  const pdb_tpi_type *rec = pdb_tpi_get_type (&ipi, item_id);
  if (rec == nullptr || rec->leaf != LF_STRING_ID || rec->data_len < 5)
    return false;

  const char *suffix = pdb_extract_string (rec->data + 4,
					  rec->data + rec->data_len);
  if (suffix == nullptr)
    return false;

  active.push_back (item_id);
  uint32_t substrings = read_u32 (rec->data);
  if (substrings != 0)
    {
      const pdb_tpi_type *list = pdb_tpi_get_type (&ipi, substrings);
      if (list == nullptr || list->leaf != LF_SUBSTR_LIST || list->data_len < 4)
	return false;
      uint32_t count = read_u32 (list->data);
      if (count > (list->data_len - 4) / 4)
	return false;
      for (uint32_t index = 0; index < count; ++index)
	if (!pdb_ipi_append_string (ipi, read_u32 (list->data + 4 + index * 4),
            name, active, remaining))
	  return false;
    }
  active.pop_back ();

  size_t length = strlen (suffix);
  if (length > UINT16_MAX - name.size ())
    return false;
  name.append (suffix, length);
  return true;
}

/* See pdb-internal.h.  */

bool
pdb_ipi_lookup_inlinee (pdb_per_objfile *pdb, uint32_t item_id,
			std::string *name, uint32_t *sig_ti)
{
  if (name != nullptr)
    name->clear ();

  const pdb_tpi_type *rec = pdb_tpi_get_type (&pdb->ipi, item_id);
  if (rec == nullptr)
    return false;

  uint32_t type_offs;
  uint32_t name_offs;
  switch (rec->leaf)
    {
    case LF_FUNC_ID:
      type_offs = LF_FUNC_ID_TYPE_OFFS;
      name_offs = LF_FUNC_ID_NAME_OFFS;
      break;
    case LF_MFUNC_ID:
      type_offs = LF_MFUNC_ID_TYPE_OFFS;
      name_offs = LF_MFUNC_ID_NAME_OFFS;
      break;
    default:
      return false;
    }

  if (rec->data_len < name_offs)
    return false;

  if (sig_ti != nullptr)
    *sig_ti = read_u32 (rec->data + type_offs);
  if (name != nullptr)
    {
      const char *leaf = pdb_extract_string (rec->data + name_offs,
					    rec->data + rec->data_len);
      if (leaf == nullptr)
	return false;

      std::string qualified;
      uint32_t scope_id = read_u32 (rec->data);
      if (rec->leaf == LF_MFUNC_ID)
	{
	  const char *scope = pdb_tpi_tag_name (&pdb->tpi, scope_id);
	  if (scope == nullptr)
	    return false;
	  qualified.assign (scope);
	}
      else if (scope_id != 0)
	{
	  std::vector<uint32_t> active;
	  uint32_t remaining = UINT16_MAX;
	  if (!pdb_ipi_append_string (pdb->ipi, scope_id, qualified, active,
				     remaining))
	    return false;
	}
      if (!qualified.empty ())
	qualified.append ("::");
      qualified.append (leaf);
      *name = std::move (qualified);
    }
  return true;
}

/* GDB type construction.  */

/* Group of overloaded methods with the same name.  Used to collect all
   overloads in an LF_METHODLIST under a single name entry.  */
struct pdb_fn_group
{
  const char *name;
  std::vector<fn_field> methods;
};

/* Result of parsing an LF_FIELDLIST: the four buckets plus the
   vtable details.  */
struct pdb_fieldlist_result
{
  std::vector<field> baseclasses;
  std::vector<field> fields;
  std::vector<decl_field> nested_types;
  std::vector<pdb_fn_group> fn_groups;
  bool has_vfptr = false;
  struct type *vfptr_type = nullptr;
};

/* Forward declarations.  */

static pdb_fieldlist_result pdb_tpi_parse_fieldlist
  (pdb_per_objfile *pdb, bool is_enum, uint32_t fieldlist_ti);
static void pdb_apply_fieldlist (pdb_per_objfile *pdb, type *type,
				 const pdb_fieldlist_result &result);
static void pdb_complete_deferred_structs (pdb_per_objfile *pdb);

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

/* Return a new TYPE_CODE_ERROR type of SIZE bytes named NAME, for a record
   whose size is known but whose value GDB cannot interpret.  */

static type *
pdb_tpi_make_opaque (pdb_per_objfile *pdb, uint32_t size, const char *name)
{
  type_allocator alloc (pdb->objfile, language_c);
  return alloc.new_type (TYPE_CODE_ERROR, size * TARGET_CHAR_BIT, name);
}

/* Resolve a built-in type index to a GDB type.
   Simple type indices are < 0x1000 and encode:
     bits 0-7  : type kind
     bits 8-10 : pointer mode (bit 11 is reserved).
   Direct types use GDB builtins.  Pointer results and pointer-size failures
   are cached by TI.  */

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
      pdb_warning ("Unknown simple type 0x%02x in TI 0x%04x", kind, type_idx);
      return pdb_tpi_get_unsupported_type (pdb);
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
    default:
      /* Legacy modes (NPTR, FPTR, HPTR) — no explicit size, assume arch.  */
      break;
    }
  if (encoded_size != 0 && encoded_size != arch_ptr_size)
    {
      pdb_warning ("simple-type pointer TI 0x%04x has mode 0x%x (size %d)"
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

/* LF_MODIFIER — apply the recorded const and volatile qualifiers.  */

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
    base = make_cv_type (is_const, is_volatile, base);

  return *base;
}

/* LF_POINTER — construct a pointer, reference or member-pointer type.  */

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
    case CV_PTRMODE_PMEM:
    case CV_PTRMODE_PMFUNC:
      {
	/* Pointer to member.  The containing-class type index follows the
	   fixed part of the record.  */
	if (rec->data_len < LF_PTR_PMCLASS_OFFS + 4)
	  {
	    pdb_warning ("LF_POINTER: pointer-to-member missing pmclass");
	    result = lookup_pointer_type (pointee);
	    break;
	  }

	if (ptrmode == CV_PTRMODE_PMFUNC)
	  {
	    /* Member function pointer (LF_MFUNCTION method type).  */
	    result = lookup_methodptr_type (pointee);
	  }
	else
	  {
	    /* Data member pointer - need to supply the class explicitly.  */
	    uint32_t pmclass_ti = read_u32 (rec->data + LF_PTR_PMCLASS_OFFS);
	    type *domain = &pdb_tpi_resolve_type (pdb, pmclass_ti);
	    result = lookup_memberptr_type (pointee, domain);
	  }
	break;
      }
    case CV_PTRMODE_POINTER:
    default:
      result = lookup_pointer_type (pointee);
      break;
    }

  /* The record gives the pointer's size.  A data member pointer of 4 bytes
     is a plain offset, which GDB reads at the type's length.  Any other
     size that differs from GDB's layout holds a narrower address or the
     adjustment fields of another inheritance model, which GDB cannot
     interpret; keep only its size, so that sizeof, array strides and
     member offsets stay right.  */
  uint32_t cv_size = cv_ptr_size (attr);
  if (cv_size != 0 && cv_size != result->length ())
    {
      if (result->code () == TYPE_CODE_MEMBERPTR && cv_size == 4)
	result->set_length (cv_size);
      else
	result = pdb_tpi_make_opaque (pdb, cv_size,
				      "<unsupported PDB pointer layout>");
    }

  if (is_const || is_volatile)
    result = make_cv_type (is_const, is_volatile, result);

  return *result;
}

/* LF_ARRAY  */

static type &
pdb_tpi_make_array (pdb_per_objfile *pdb, const pdb_tpi_type *rec)
{
  if (!pdb_check_record_size (rec, LF_ARRAY_MIN_SIZE, "LF_ARRAY"))
    return pdb_tpi_get_unsupported_type (pdb);

  uint32_t elem_ti = read_u32 (rec->data + LF_ARRAY_ELEMTYPE_OFFS);
  uint32_t idx_ti = read_u32 (rec->data + LF_ARRAY_IDXTYPE_OFFS);

  auto *elem_type = &pdb_tpi_resolve_type (pdb, elem_ti);
  auto *idx_type = &pdb_tpi_resolve_type (pdb, idx_ti);

  /* Trailing data: numeric leaf for total size in bytes, then a
     NUL-terminated name (may be empty).  */
  const gdb_byte *d = rec->data + LF_ARRAY_DATA_OFFS;
  const gdb_byte *end = rec->data + rec->data_len;
  uint32_t data_len = rec->data_len > LF_ARRAY_DATA_OFFS
			? rec->data_len - LF_ARRAY_DATA_OFFS
			: 0;

  uint64_t total_size = 0;
  uint32_t consumed = 0;
  if (data_len > 0)
    consumed = pdb_cv_read_numeric (d, data_len, &total_size);

  if (consumed == 0 && data_len > 0)
    pdb_dbg_printf ("LF_ARRAY: failed to read numeric leaf for array size");

  /* Optional trailing name (may be empty).  */
  const char *name = nullptr;
  if (consumed > 0)
    name = pdb_extract_string (d + consumed, end);

  uint64_t elem_size = elem_type->length ();

  uint64_t num_elements = 0;
  if (elem_size > 0)
    {
      num_elements = total_size / elem_size;
      if (total_size % elem_size != 0)
	pdb_warning ("LF_ARRAY: total size %" PRIu64 " not a multiple of"
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
  array_type->set_length (total_size);

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

/* Resolve argument list from LF_ARGLIST record and populate func_type fields.
   EXTRA_SLOTS reserves leading field slots (e.g. 1 for an implicit 'this'
   parameter that the caller fills separately).  Arglist entries are placed
   starting at field index EXTRA_SLOTS.  */

static void
pdb_tpi_resolve_arglist (pdb_per_objfile *pdb, type *func_type,
			 uint32_t arglist_ti, uint32_t extra_slots = 0)
{
  /* An unusable arglist still has to leave EXTRA_SLOTS allocated: the
     caller fills those slots itself.  */
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

  uint32_t total = extra_slots + argc;
  if (total == 0)
    return;

  func_type->alloc_fields (total);
  for (uint32_t i = 0; i < argc; i++)
    {
      auto arg_ti = read_u32 (rec->data + LF_ARGLIST_ARGS_OFFS + i * 4);
      auto *arg_type = &pdb_tpi_resolve_type (pdb, arg_ti);
      func_type->field (extra_slots + i).set_type (arg_type);
    }
}

/* LF_PROCEDURE  */

static type &
pdb_tpi_make_procedure (pdb_per_objfile *pdb, const pdb_tpi_type *rec)
{
  if (!pdb_check_record_size (rec, LF_PROC_SIZE, "LF_PROCEDURE"))
    return pdb_tpi_get_unsupported_type (pdb);

  auto ret_ti = read_u32 (rec->data + LF_PROC_RVTYPE_OFFS);
  auto arglist_ti = read_u32 (rec->data + LF_PROC_ARGLIST_OFFS);

  auto *ret_type = &pdb_tpi_resolve_type (pdb, ret_ti);
  auto *func_type = lookup_function_type (ret_type);

    /* Treat the recorded argument list as a prototype.  For TYPE_CODE_FUNC,
      infcall applies default promotions beyond the declared fields
      independently of is_prototyped, including a variadic tail.  */
  func_type->set_is_prototyped (true);

  pdb_tpi_resolve_arglist (pdb, func_type, arglist_ti);
  return *func_type;
}

/* LF_MFUNCTION — Member function.
   Reads class_ti, this_ti, ret_ti and arglist_ti from the record.
   Builds a TYPE_CODE_METHOD with self_type set to the containing class
  and 'this' as artificial field 0.  Static methods (this_ti == 0)
   get no 'this' parameter.  */

static type &
pdb_tpi_make_mfunction (pdb_per_objfile *pdb, const pdb_tpi_type *rec)
{
  if (!pdb_check_record_size (rec, LF_MFUNC_SIZE, "LF_MFUNCTION"))
    return pdb_tpi_get_unsupported_type (pdb);

  auto ret_ti = read_u32 (rec->data + LF_MFUNC_RVTYPE_OFFS);
  auto class_ti = read_u32 (rec->data + LF_MFUNC_CLASSTYPE_OFFS);
  auto this_ti = read_u32 (rec->data + LF_MFUNC_THISTYPE_OFFS);
  auto arglist_ti = read_u32 (rec->data + LF_MFUNC_ARGLIST_OFFS);

  auto *ret_type = &pdb_tpi_resolve_type (pdb, ret_ti);
  auto *class_type = &pdb_tpi_resolve_type (pdb, class_ti);
  auto *func_type = lookup_method_type (class_type, ret_type);

  func_type->set_is_prototyped (true);

  /* For non-static methods (this_ti != 0), reserve slot 0 for the
     implicit 'this' pointer and fill arglist starting at slot 1.  */
  bool has_this = (this_ti != 0);
  pdb_tpi_resolve_arglist (pdb, func_type, arglist_ti, has_this ? 1 : 0);

  if (has_this)
    {
      auto *this_type = &pdb_tpi_resolve_type (pdb, this_ti);
      func_type->field (0).set_type (this_type);
      func_type->field (0).set_is_artificial (true);

      const pdb_tpi_type *this_rec = pdb_tpi_get_type (&pdb->tpi, this_ti);
      if (this_rec != nullptr && this_rec->leaf == LF_POINTER
	  && this_rec->data_len >= LF_POINTER_SIZE)
	{
	  uint32_t attr = read_u32 (this_rec->data + LF_POINTER_ATTR_OFFS);
	  if ((attr & LF_POINTER_ATTR_ISRREF) != 0)
	    pdb->tpi.method_ref_qualifier[func_type] = " &&";
	  else if ((attr & LF_POINTER_ATTR_ISLREF) != 0)
	    pdb->tpi.method_ref_qualifier[func_type] = " &";
	}
    }

  return *func_type;
}

/* See pdb-internal.h.  */

const char *
pdb_method_ref_qualifier (const pdb_per_objfile *pdb, const type *method_type)
{
  auto it = pdb->tpi.method_ref_qualifier.find (method_type);
  return it == pdb->tpi.method_ref_qualifier.end () ? "" : it->second;
}

/* Allocate a struct/union/enum and cache it before resolving its fields,
   so a reference back to it finds the cached type.  A forward record, which
   pdb_tpi_resolve_type reaches only for a tag the TPI never defines, becomes
   a stub.  An enum is populated at once, since enumerators reference no
   types.  A struct or union is left as a shell: its field list is deferred
   and read by pdb_complete_deferred_structs, or earlier by
   pdb_complete_struct when it is a base class.  */

static type *
pdb_tpi_init_compound (pdb_per_objfile *pdb, enum type_code code,
		       const pdb_compound_fields *f,
		       uint32_t type_idx [[maybe_unused]])
{
  type_allocator alloc (pdb->objfile, language_c);
  type *type = alloc.new_type ();

  type->set_code (code);

  if (f->name[0] != '\0')
    type->set_name (pdb_canonical_name (pdb, f->name));

  type->set_length (f->byte_size);

  /* Cache the type before parsing its fields.  */
  if (pdb->tpi.type_cache != nullptr)
    pdb->tpi.type_cache[type_idx] = type;

  /* Forward reference — mark as stub.  */
  if (f->property & CV_PROP_FWDREF)
    {
      type->set_is_stub (true);
      return type;
    }

  if (f->fieldlist_ti == 0)
    return type;

  if (code == TYPE_CODE_ENUM)
    {
      pdb_fieldlist_result parse
	= pdb_tpi_parse_fieldlist (pdb, true, f->fieldlist_ti);
      pdb_apply_fieldlist (pdb, type, parse);
    }
  else
    {
      pdb->tpi.deferred_structs.emplace (type, f->fieldlist_ti);
      pdb->tpi.deferred_struct_order.push_back (type);
    }

  return type;
}

/* LF_STRUCTURE / LF_CLASS.  */

static type &
pdb_tpi_make_struct (pdb_per_objfile *pdb, const pdb_tpi_type *rec,
		     uint32_t type_idx)
{
  auto f = pdb_parse_tagged_record (rec, LF_STRUCT_MIN_SIZE,
				    LF_STRUCT_FIELDLIST_OFFS,
				    LF_STRUCT_DATA_OFFS, "LF_STRUCTURE/CLASS");
  if (!f)
    return pdb_tpi_get_unsupported_type (pdb);

  type *type = pdb_tpi_init_compound (pdb, TYPE_CODE_STRUCT, &*f, type_idx);

  if (rec->leaf == LF_CLASS)
    type->set_is_declared_class (true);

  return *type;
}

/* LF_UNION.  */

static type &
pdb_tpi_make_union (pdb_per_objfile *pdb, const pdb_tpi_type *rec,
		    uint32_t type_idx)
{
  auto f = pdb_parse_tagged_record (rec, LF_UNION_MIN_SIZE,
				    LF_UNION_FIELDLIST_OFFS,
				    LF_UNION_DATA_OFFS, "LF_UNION");
  if (!f)
    return pdb_tpi_get_unsupported_type (pdb);

  return *pdb_tpi_init_compound (pdb, TYPE_CODE_UNION, &*f, type_idx);
}

/* LF_ENUM.  */

static type &
pdb_tpi_make_enum (pdb_per_objfile *pdb, const pdb_tpi_type *rec,
		   uint32_t type_idx)
{
  if (!pdb_check_record_size (rec, LF_ENUM_MIN_SIZE, "LF_ENUM"))
    return pdb_tpi_get_unsupported_type (pdb);

  /* Get underlying type.  */
  uint32_t utype_ti = read_u32 (rec->data + LF_ENUM_UTYPE_OFFS);
  type *enum_type = &pdb_tpi_resolve_type (pdb, utype_ti);

  const char *name = pdb_extract_string (rec->data + LF_ENUM_NAME_OFFS,
					 rec->data + rec->data_len);
  if (name == nullptr)
    name = "";

  pdb_compound_fields f;
  f.count = read_u16 (rec->data + LF_ENUM_COUNT_OFFS);
  f.property = read_u16 (rec->data + LF_ENUM_PROPERTY_OFFS);
  f.fieldlist_ti = read_u32 (rec->data + LF_ENUM_FIELDLIST_OFFS);
  /* Byte size comes from the underlying integer type.  */
  f.byte_size = enum_type->length ();
  f.name = name;

  type *type = pdb_tpi_init_compound (pdb, TYPE_CODE_ENUM, &f, type_idx);

  type->set_target_type (enum_type);
  type->set_is_unsigned (enum_type->is_unsigned ());

  return *type;
}

/* Build the GDB type for compound TPI record REC.
   Dispatches to the pdb_tpi_make_* helpers; non-type leaves resolve to
   builtin void and unknown leaves to the cached "<unsupported PDB type>".
   Never returns a null type.  */

static type &
pdb_tpi_build_type (pdb_per_objfile *pdb, const pdb_tpi_type *rec,
		    uint32_t type_idx)
{
  switch (rec->leaf)
    {
    case LF_MODIFIER:
      return pdb_tpi_make_modifier (pdb, rec);
    case LF_PROCEDURE:
      return pdb_tpi_make_procedure (pdb, rec);
    case LF_MFUNCTION:
      return pdb_tpi_make_mfunction (pdb, rec);
    case LF_POINTER:
      return pdb_tpi_make_pointer (pdb, rec);
    case LF_ARRAY:
      return pdb_tpi_make_array (pdb, rec);
    case LF_BITFIELD:
      return pdb_tpi_make_bitfield (pdb, rec);
    case LF_STRUCTURE:
    case LF_CLASS:
      return pdb_tpi_make_struct (pdb, rec, type_idx);
    case LF_UNION:
      return pdb_tpi_make_union (pdb, rec, type_idx);
    case LF_ENUM:
      return pdb_tpi_make_enum (pdb, rec, type_idx);
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

/* Recursive type resolver with caching.
   Compound types dispatch to pdb_tpi_make_* helpers.
   Unsupported leaf types return the cached "<unsupported PDB type>".
   A forward record resolves to its definition when the TPI has one.
   Called while a field list is being read, it can return a struct or union
   shell whose members are deferred; the outermost call reads the members of
   every deferred shell before returning.  */

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

  /* A forward record carries only the tag name.  Resolve it as the
     definition selected for that name by pdb_tpi_build_tagged_name_cache,
     so the forward record and the definition share one GDB type.  The
     table is indexed by TI - type_idx_begin and is empty when the TPI has
     no records.  */
  if (type_idx >= tpi.type_idx_begin && !tpi.fwdref_definition.empty ())
    if (uint32_t def = tpi.fwdref_definition[type_idx - tpi.type_idx_begin];
	def != 0)
      type_idx = def;

  type *result = tpi.type_cache[type_idx];
  if (result == nullptr)
    {
      const pdb_tpi_type *rec = pdb_tpi_get_type (&tpi, type_idx);
      if (rec == nullptr)
	return pdb_tpi_get_unsupported_type (pdb);

      /* A record may be reached again while its build is in progress only
	 through a field list read in between.  */
      std::pair<uint32_t, int> build (type_idx, tpi.struct_read_depth);
      if (std::find (tpi.records_in_progress.begin (),
		     tpi.records_in_progress.end (), build)
	  != tpi.records_in_progress.end ())
	{
	  pdb_complaint ("type index 0x%x refers to itself", type_idx);
	  return pdb_tpi_get_unsupported_type (pdb);
	}

      tpi.records_in_progress.push_back (build);
      auto pop = make_scope_exit ([&] ()
	{
	  tpi.records_in_progress.pop_back ();
	});

      result = &pdb_tpi_build_type (pdb, rec, type_idx);
      tpi.type_cache[type_idx] = result;
    }

  if (tpi.struct_read_depth == 0)
    pdb_complete_deferred_structs (pdb);

  return *result;
}

/* Struct, union and enum members.  */

/* Set field accessibility from CV_fldattr_t access bits.  */

static void
pdb_set_field_accessibility (field *fp, uint16_t attr)
{
  switch (attr & CV_ACCESS_MASK)
    {
    case CV_ACCESS_PRIVATE:
      fp->set_accessibility (accessibility::PRIVATE);
      break;
    case CV_ACCESS_PROTECTED:
      fp->set_accessibility (accessibility::PROTECTED);
      break;
    case CV_ACCESS_PUBLIC:
      /* Public is the default accessibility.  */
      break;
    default:
      break;
    }
}

/* Search for an existing method group by NAME.  If found, return a reference
   to it.  If not found, create a new group with that name and return it.
   Used when parsing LF_ONEMETHOD and LF_METHOD sub-records: multiple
   overloads of the same function name are collected into one group.  */

static pdb_fn_group &
pdb_find_or_add_fn_group (std::vector<pdb_fn_group> &groups, const char *name)
{
  for (auto &g : groups)
    if (strcmp (g.name, name) == 0)
      return g;

  auto &group = groups.emplace_back ();
  group.name = name;
  return group;
}

/* Build an fn_field from a CodeView method's attributes (ATTR) and type
   index (TYPE_TI).  VBASEOFF is the vtable byte offset, carried by the
   sub-record only for introducing virtuals (INTRO/PUREINTRO).
   Sets type, accessibility, and voffset.

   Note on voffset:
   Here we don't know the virtual slot for an override method
   (VIRTUAL/PUREVIRT) because the CodeView stores a virtual's vtable slot
   only on its introducing declaration (the INTRO/PUREINTRO).  The override
   should use the slot of the virtual it overrides.  Here we use the placeholder
   value of voffset=2 so that the encoding counts as "virtual".  A later fix-up
   pass finds the real slot: the introducing method lives in a base class,
   reached through the derived class's LF_BCLASS records, and its fn_field is
   matched by name and parameter list.

   Note on physname:
   Physname is initialised to "".  pdb_add_method_overload stores the bare
   name for non-friend methods; pdb_apply_fieldlist later qualifies it per
   owner so lookup_symbol can resolve symbols.  */

static fn_field
pdb_fill_fn_field (pdb_per_objfile *pdb, uint16_t attr, uint32_t type_ti,
		   uint32_t vbaseoff = 0)
{
  fn_field fnp {};
  fnp.type = &pdb_tpi_resolve_type (pdb, type_ti);
  fnp.physname = "";

  uint16_t access = attr & CV_ACCESS_MASK;
  switch (access)
    {
    case CV_ACCESS_PRIVATE:
      fnp.accessibility = accessibility::PRIVATE;
      break;

    case CV_ACCESS_PROTECTED:
      fnp.accessibility = accessibility::PROTECTED;
      break;

    default:
      fnp.accessibility = accessibility::PUBLIC;
      break;
    }

  uint16_t mprop = CV_MPROP (attr);
  switch (mprop)
    {
    case CV_MPROP_STATIC:
      fnp.voffset = VOFFSET_STATIC;
      break;
    case CV_MPROP_INTRO:
    case CV_MPROP_PUREINTRO:
      {
	/* Convert vbaseoff to a slot index and add 2 to get virtual.  */
	int ptr_size = gdbarch_ptr_bit (pdb->objfile->arch ()) / 8;
	fnp.voffset = (vbaseoff / ptr_size) + 2;
	break;
      }
    case CV_MPROP_VIRTUAL:
    case CV_MPROP_PUREVIRT:
      /* Keep the virtual placeholder unless base-method lookup finds a
	 matching slot after fieldlist parsing.  */
      fnp.voffset = 2;
      break;
    default:
      fnp.voffset = 0;
      break;
    }

  return fnp;
}

/* Build "EnclosingTag::name" for method-symbol lookup.  Return the bare NAME
   if ENCLOSING_NAME is null or empty.  Qualified names use objfile storage;
   the fallback retains the caller's string.  */

static const char *
pdb_build_method_physname (pdb_per_objfile *pdb, const char *enclosing_name,
			   const char *name)
{
  if (enclosing_name == nullptr || *enclosing_name == '\0')
    return name;
  return obconcat (&pdb->objfile->objfile_obstack, enclosing_name, "::", name,
		   (char *) nullptr);
}

/* Format the parameter suffix used to distinguish overloaded physnames.
   Artificial fields, including 'this', are omitted.  A non-variadic method
   with no explicit parameters keeps the bare name.  An ellipsis-only method
   gets "(...)" when it has an artificial field; the static form, with no
   fields, keeps the bare name.  A ref-qualified method always gets its
   parameter list followed by the qualifier, spelled as pdb_func_sym names
   the procedure symbol.  */

static std::string
pdb_method_param_signature (const pdb_per_objfile *pdb, const fn_field &m)
{
  struct type *ftype = m.type;
  if (ftype == nullptr)
    return "";

  const char *ref_qualifier = pdb_method_ref_qualifier (pdb, ftype);

  int named = 0;
  for (const auto &f : ftype->fields ())
    if (!f.is_artificial ())
      named++;

  if (named == 0 && *ref_qualifier == '\0')
    {
      /* An ellipsis-only method is spelled only when the record carried a
	 'this' slot; the static form keeps the bare name.  */
      if (ftype->has_varargs () && ftype->num_fields () != 0)
	return "(...)";
      return "";
    }

  /* Use the same linkage-name mode and raw options as pdb_func_sym for
     non-empty parameter lists.  Raw options bypass name canonicalization,
     but type printing can still call check_typedef.  The method table must
     remain unpublished until all groups are initialized.  */
  string_file buf;
  c_type_print_args (ftype, &buf, 1, language_cplus, &type_print_raw_options);
  buf.puts (ref_qualifier);
  return buf.release ();
}

/* Build one method overload from ATTR/TYPE_TI/VBASEOFF and append it to GRP.
   A member's physname is the bare method name; pdb_apply_fieldlist later
  qualifies it to "Tag::name" for lookup of the procedure symbol by name.
  A friend's physname stays empty and is not qualified.  */

static void
pdb_add_method_overload (pdb_per_objfile *pdb, pdb_fn_group &grp,
			 const char *name, uint16_t attr, uint32_t type_ti,
			 uint32_t vbaseoff)
{
  fn_field fnp = pdb_fill_fn_field (pdb, attr, type_ti, vbaseoff);
  if (CV_MPROP (attr) != CV_MPROP_FRIEND)
    fnp.physname = name;
  grp.methods.push_back (fnp);
}

/* LF_FIELDLIST sub-record handlers.

   Each pdb_parse_lf_* function handles one sub-record leaf type from
   an LF_FIELDLIST.  They read fields from the raw sub-record at P,
   resolve any type indices, and push the result into the appropriate
   output vector (fields, baseclasses, nested_types, or fn_groups).
   pdb_fieldlist_skip_record checks each record's extent before dispatch.  */

static void pdb_complete_struct (pdb_per_objfile *pdb, type *type);

/* LF_MEMBER — non-static data member of a struct/union.  */

static void
pdb_parse_lf_member (pdb_per_objfile *pdb, const gdb_byte *p,
		     const gdb_byte *end, std::vector<field> &fields)
{
  uint16_t attr = read_u16 (p + LF_MEMBER_ATTR_OFFS);
  uint32_t type_ti = read_u32 (p + LF_MEMBER_TYPE_OFFS);
  const gdb_byte *d = p + LF_MEMBER_DATA_OFFS;

  /* Decode the member's byte offset from the numeric leaf.  */
  uint64_t offset_val;
  auto nr = pdb_cv_read_numeric (d, (uint32_t) (end - d), &offset_val);
  if (nr == 0)
    {
      pdb_warning ("LF_MEMBER: truncated numeric leaf");
      return;
    }
  d += nr;
  const char *name = pdb_extract_string (d, end);
  if (name == nullptr)
    {
      pdb_warning ("LF_MEMBER: missing NUL-terminated name");
      return;
    }

  type *ftype = &pdb_tpi_resolve_type (pdb, type_ti);
  field f {};
  f.set_name (name);
  f.set_type (ftype);
  f.set_loc_bitpos (offset_val * 8);

  /* When type_ti points to an LF_BITFIELD record, fetch the bit-width and
     bit-position, then update the field's bitpos and size.  */
  if (const pdb_tpi_type *member_type_rec = pdb_tpi_get_type (&pdb->tpi,
							      type_ti);
      member_type_rec != nullptr && member_type_rec->leaf == LF_BITFIELD)
    {
      if (pdb_check_record_size (member_type_rec, LF_BITFIELD_SIZE,
				 "LF_BITFIELD (member)"))
	{
	  uint8_t bf_width = read_u8 (member_type_rec->data
				      + LF_BITFIELD_LENGTH_OFFS);
	  uint8_t bf_pos = read_u8 (member_type_rec->data
				    + LF_BITFIELD_POSITION_OFFS);
	  if (bf_width > 0)
	    {
	      f.set_bitsize (bf_width);
	      f.set_loc_bitpos (offset_val * 8 + bf_pos);
	    }
	}
    }

  pdb_set_field_accessibility (&f, attr);
  fields.push_back (f);
}

/* LF_ENUMERATE — one enumerator constant in an enum type.
   Field layout defined by LF_ENUMERATE_* offsets in pdb-internal.h.  */

static void
pdb_parse_lf_enumerate (pdb_per_objfile *pdb, const gdb_byte *p,
			const gdb_byte *end, std::vector<field> &fields)
{
  uint16_t attr = read_u16 (p + LF_ENUMERATE_ATTR_OFFS);
  const gdb_byte *d = p + LF_ENUMERATE_DATA_OFFS;

  uint64_t val;
  auto nr = pdb_cv_read_numeric (d, (uint32_t) (end - d), &val);
  if (nr == 0)
    {
      pdb_warning ("LF_ENUMERATE: truncated numeric leaf");
      return;
    }
  d += nr;
  const char *name = pdb_extract_string (d, end);
  if (name == nullptr)
    {
      pdb_warning ("LF_ENUMERATE: missing NUL-terminated name");
      return;
    }

  field f {};
  f.set_name (name);
  f.set_loc_enumval ((LONGEST) val);
  pdb_set_field_accessibility (&f, attr);
  fields.push_back (f);
}

/* LF_BCLASS — direct (non-virtual) base class.
   Field layout defined by LF_BCLASS_* offsets in pdb-internal.h.
   The base's members are read now: override slots are recovered from its
   method table.  */

static void
pdb_parse_lf_bclass (pdb_per_objfile *pdb, const gdb_byte *p,
		     const gdb_byte *end, std::vector<field> &baseclasses)
{
  uint16_t attr = read_u16 (p + LF_BCLASS_ATTR_OFFS);
  uint32_t base_ti = read_u32 (p + LF_BCLASS_TYPE_OFFS);
  const gdb_byte *d = p + LF_BCLASS_DATA_OFFS;

  uint64_t offset;
  pdb_cv_read_numeric (d, (uint32_t) (end - d), &offset);

  type *base_type = &pdb_tpi_resolve_type (pdb, base_ti);
  pdb_complete_struct (pdb, base_type);
  field f {};
  f.set_type (base_type);
  f.set_name (base_type->name () ? base_type->name () : "");
  f.set_loc_bitpos (offset * 8);
  pdb_set_field_accessibility (&f, attr);
  baseclasses.push_back (f);
}

/* LF_STMEMBER — static data member.

   This record carries only the unqualified name (e.g. "count").  GDB's
   value_static_field () looks up the underlying global by the field's
   physname, which must be the fully qualified "Tag::name" that MSVC emits as
   an S_GDATA32 / S_LDATA32 record (see pdb_var_sym in pdb-read-symbols.c).
   The bare name is stored as physname here and qualified with the owning tag
  in pdb_apply_fieldlist.  set_loc_physname selects the static-field
  location kind.  */

static void
pdb_parse_lf_stmember (pdb_per_objfile *pdb, const gdb_byte *p,
		       const gdb_byte *end, std::vector<field> &fields)
{
  uint16_t attr = read_u16 (p + LF_STMEMBER_ATTR_OFFS);
  uint32_t type_ti = read_u32 (p + LF_STMEMBER_TYPE_OFFS);
  const char *name = pdb_extract_string (p + LF_STMEMBER_NAME_OFFS, end);
  if (name == nullptr)
    {
      pdb_warning ("LF_STMEMBER: missing NUL-terminated name");
      return;
    }

  type *ftype = &pdb_tpi_resolve_type (pdb, type_ti);
  field f {};
  f.set_name (name);
  f.set_type (ftype);
  f.set_loc_physname (name);

  pdb_set_field_accessibility (&f, attr);
  fields.push_back (f);
}

/* LF_NESTTYPE — a nested type or a class-scope alias to another type.
  The record has no access attributes; this reader uses PUBLIC.  */

static void
pdb_parse_lf_nesttype (pdb_per_objfile *pdb, const gdb_byte *p,
		       const gdb_byte *end,
		       std::vector<decl_field> &nested_types)
{
  uint32_t nested_ti = read_u32 (p + LF_NESTTYPE_TYPE_OFFS);
  const char *name = pdb_extract_string (p + LF_NESTTYPE_NAME_OFFS, end);
  if (name == nullptr)
    {
      pdb_warning ("LF_NESTTYPE: missing NUL-terminated name");
      return;
    }

  type *nested_type = &pdb_tpi_resolve_type (pdb, nested_ti);
  decl_field df;
  df.name = name;
  df.type = nested_type;
  df.accessibility = accessibility::PUBLIC;
  nested_types.push_back (df);
}

/* LF_ONEMETHOD — a single overload of a member function.  */

static void
pdb_parse_lf_onemethod (pdb_per_objfile *pdb, const gdb_byte *p,
			const gdb_byte *end,
			std::vector<pdb_fn_group> &fn_groups)
{
  uint16_t attr = read_u16 (p + LF_ONEMETHOD_ATTR_OFFS);
  uint32_t type_ti = read_u32 (p + LF_ONEMETHOD_TYPE_OFFS);
  const gdb_byte *d = p + LF_ONEMETHOD_DATA_OFFS;

  /* INTRO/PUREINTRO virtual methods (introducing ones) carry an extra
     uint32_t vbaseoff (vtable slot byte offset) before the name.  */
  uint32_t vbaseoff = 0;
  if (CV_MPROP_HAS_VBASEOFF (CV_MPROP (attr)))
    {
      if (d + 4 > end)
	{
	  pdb_warning ("LF_ONEMETHOD: truncated vbaseoff");
	  return;
	}
      vbaseoff = read_u32 (d);
      d += 4;
    }
  const char *name = pdb_extract_string (d, end);
  if (name == nullptr)
    {
      pdb_warning ("LF_ONEMETHOD: missing NUL-terminated name");
      return;
    }

  pdb_fn_group &grp = pdb_find_or_add_fn_group (fn_groups, name);
  pdb_add_method_overload (pdb, grp, name, attr, type_ti, vbaseoff);
}

/* LF_METHOD — overloaded member function (multiple overloads).
   Reads the referenced LF_METHODLIST (layout: LF_MLIST_* offsets) and
   adds each overload to the same fn_group.  */

static void
pdb_parse_lf_method (pdb_per_objfile *pdb, const gdb_byte *p,
		     const gdb_byte *end, std::vector<pdb_fn_group> &fn_groups)
{
  uint16_t mcount = read_u16 (p + LF_METHOD_COUNT_OFFS);
  uint32_t mlist_ti = read_u32 (p + LF_METHOD_MLIST_OFFS);
  const char *name = pdb_extract_string (p + LF_METHOD_NAME_OFFS, end);
  if (name == nullptr)
    {
      pdb_warning ("LF_METHOD: missing NUL-terminated name");
      return;
    }

  const pdb_tpi_type *ml = pdb_tpi_get_type (&pdb->tpi, mlist_ti);
  if (ml == nullptr || ml->leaf != LF_METHODLIST)
    return;

  pdb_fn_group &grp = pdb_find_or_add_fn_group (fn_groups, name);

  /* Walk the LF_METHODLIST entries.  Each is 8 bytes (attr + pad + type_ti),
     optionally followed by 4 bytes (vbaseoff) for introducing virtuals.  */
  const gdb_byte *mp = ml->data;
  const gdb_byte *mend = ml->data + ml->data_len;
  for (uint16_t i = 0; i < mcount && mp + LF_MLIST_ENTRY_SIZE <= mend; i++)
    {
      uint16_t mattr = read_u16 (mp + LF_MLIST_ATTR_OFFS);
      uint32_t mtype_ti = read_u32 (mp + LF_MLIST_TYPE_OFFS);
      mp += LF_MLIST_ENTRY_SIZE;

      uint32_t vbaseoff = 0;
      if (CV_MPROP_HAS_VBASEOFF (CV_MPROP (mattr)))
	{
	  if (mp + 4 > mend)
	    break;
	  vbaseoff = read_u32 (mp);
	  mp += 4;
	}

      pdb_add_method_overload (pdb, grp, name, mattr, mtype_ti, vbaseoff);
    }
}

/* LF_VBCLASS — a direct virtual base class.  An indirect virtual base
  (LF_IVBCLASS) is inherited through one of these and is not a field of its
  own, so only this leaf reaches here.
  The skip helper validates the vbptr offset and vbtable index leaves, but
  this handler retains only the base type, access and virtual flag.  Runtime
  access needs interpretation of the MSVC virtual-base layout.  */

static void
pdb_parse_lf_vbclass (pdb_per_objfile *pdb, const gdb_byte *p,
		      const gdb_byte *end [[maybe_unused]],
		      std::vector<field> &baseclasses)
{
  uint16_t attr = read_u16 (p + LF_VBCLASS_ATTR_OFFS);
  uint32_t base_ti = read_u32 (p + LF_VBCLASS_TYPE_OFFS);

  type *base_type = &pdb_tpi_resolve_type (pdb, base_ti);
  pdb_complete_struct (pdb, base_type);
  field f {};
  f.set_type (base_type);
  f.set_name (base_type->name () ? base_type->name () : "");
  f.set_loc_bitpos (0);
  f.set_virtual ();
  pdb_set_field_accessibility (&f, attr);
  baseclasses.push_back (f);
}

/* Compare the parameter lists of two TYPE_CODE_METHOD types to match
   an override to a base method.  Compare the ref-qualifiers and the
   cv-qualifiers of the object pointed to by 'this', but not its class
   identity.  Compare explicit parameters with types_equal; omit return
   types to allow covariant returns.  */

static bool
pdb_method_params_match (const pdb_per_objfile *pdb, struct type *a,
			 struct type *b)
{
  if (a == nullptr || b == nullptr)
    return false;

  a = check_typedef (a);
  b = check_typedef (b);
  if (a->code () != TYPE_CODE_METHOD || b->code () != TYPE_CODE_METHOD)
    return false;

  if (a->num_fields () != b->num_fields ())
    return false;

  if (strcmp (pdb_method_ref_qualifier (pdb, a),
	      pdb_method_ref_qualifier (pdb, b)) != 0)
    return false;

  /* Method cv-qualifiers belong to the object pointed to by 'this'.  */
  auto this_target = [] (struct type *m) -> struct type *
    {
      if (m->num_fields () == 0 || !m->field (0).is_artificial ())
	return nullptr;
      struct type *t = m->field (0).type ();
      if (t == nullptr || t->code () != TYPE_CODE_PTR)
	return nullptr;
      return t->target_type ();
    };
  struct type *a_this = this_target (a);
  struct type *b_this = this_target (b);
  if (a_this != nullptr && b_this != nullptr)
    {
      if (TYPE_CONST (a_this) != TYPE_CONST (b_this))
	return false;
      if (TYPE_VOLATILE (a_this) != TYPE_VOLATILE (b_this))
	return false;
    }

  /* Compare params.  Skip field(0) if 'this'.  */
  int ia = 0, ib = 0;
  if (a->num_fields () > 0 && a->field (0).is_artificial ())
    ia = 1;
  if (b->num_fields () > 0 && b->field (0).is_artificial ())
    ib = 1;

  for (; ia < a->num_fields () && ib < b->num_fields (); ia++, ib++)
    if (!types_equal (a->field (ia).type (), b->field (ib).type ()))
      return false;

  return true;
}

/* Recover the encoded voffset for NAME with parameters matching METHOD_TYPE.
  Search BASES and, recursively, their bases; return -1 if no match is found.

  CodeView records a slot on the introducing declaration (INTRO/PUREINTRO),
  not on overrides.  A base table may supply that slot directly or through
  an override already fixed up.  pdb_parse_lf_bclass and
  pdb_parse_lf_vbclass populate each base before this runs.  */

static int
pdb_find_introducer_voffset (pdb_per_objfile *pdb,
			     const std::vector<field> &bases, const char *name,
			     struct type *method_type)
{
  for (const field &b : bases)
    {
      struct type *bt = b.type ();
      if (bt == nullptr)
	continue;
      bt = check_typedef (bt);

      if (bt->code () != TYPE_CODE_STRUCT)
	continue;

      /* Search this base's own method groups for a matching virtual.  */
      for (int i = 0; i < TYPE_NFN_FIELDS (bt); i++)
	{
	  const char *gname = TYPE_FN_FIELDLIST_NAME (bt, i);
	  if (gname == nullptr || strcmp (gname, name) != 0)
	    continue;
	  fn_field *fns = TYPE_FN_FIELDLIST1 (bt, i);
	  int len = TYPE_FN_FIELDLIST_LENGTH (bt, i);
	  for (int j = 0; j < len; j++)
	    if (TYPE_FN_FIELD_VIRTUAL_P (fns, j)
		&& pdb_method_params_match (pdb, fns[j].type, method_type))
	      return fns[j].voffset;
	}

      /* Recurse into this base's own bases.  */
      int nb = TYPE_N_BASECLASSES (bt);
      if (nb > 0)
	{
	  const struct field *bf = bt->fields ().data ();
	  std::vector<field> bbases (bf, bf + nb);
	  int vo = pdb_find_introducer_voffset (pdb, bbases, name,
						method_type);
	  if (vo >= 2)
	    return vo;
	}
    }

  return -1;
}

/* Parse the LF_FIELDLIST at FIELDLIST_TI into the four buckets (data members,
   base classes, nested types, method functions) plus the vtable facts
   (HAS_VFPTR / VFPTR_TYPE).  pdb_read_struct_members, and
   pdb_tpi_init_compound for an enum, pass the result to
   pdb_apply_fieldlist to stitch onto a specific type.

   IS_ENUM selects enum vs struct/class/union sub-record handling.

   Recover virtual-override vtable slots here.  CodeView records the slot only
   on a virtual's introducing declaration; an override's record omits it and
  reaches this point with the placeholder 2.  After the walk, every virtual
  method is searched for in the bases.  A match supplies its voffset; an
  unmatched method retains its original value.

  Results without methods or a vptr are cached by fieldlist TI for sharing
  between tags.  Static names remain bare in the cached fields and are
  qualified separately for each owner.  */

static pdb_fieldlist_result
pdb_tpi_parse_fieldlist (pdb_per_objfile *pdb, bool is_enum,
			 uint32_t fieldlist_ti)
{
  pdb_fieldlist_result result;

  const pdb_tpi_type *fl = pdb_tpi_get_type (&pdb->tpi, fieldlist_ti);
  if (fl == nullptr || fl->leaf != LF_FIELDLIST)
    {
      /* Seen in the merged CRT type stream, where a class names a field list
	 that is some other record; llvm-pdbutil reads the same bytes.  An
	 empty field list is the safe answer, so do not shout about it.  */
      pdb_complaint ("LF_FIELDLIST: type index 0x%x invalid or wrong leaf",
		     fieldlist_ti);
      return result;
    }

  /* Reuse fieldlist if already in cache.  */
  auto cache_it = pdb->tpi.fieldlist_cache.find (fieldlist_ti);
  if (cache_it != pdb->tpi.fieldlist_cache.end ())
    {
      const auto &c = cache_it->second;
      result.baseclasses = c.baseclasses;
      result.fields = c.fields;
      result.nested_types = c.nested_types;
      return result;
    }

  /* Parse fieldlist.  */

  const gdb_byte *p = fl->data;
  const gdb_byte *end = fl->data + fl->data_len;

  /* Four output buckets (attached to the type at the end).  */
  std::vector<field> baseclasses;
  std::vector<field> fields;
  std::vector<decl_field> nested_types;
  std::vector<pdb_fn_group> fn_groups;

  bool has_vfptr = false;
  struct type *vfptr_type = nullptr;
  std::vector<uint32_t> visited { fieldlist_ti };

  while (p < end)
    {
      /* Sub-records are 4-byte aligned.  Also check the
	 padding bytes are LF_PAD markers (0xf3, 0xf2 or 0xf1).  */
      const gdb_byte *aligned = align_up (p, 4);
      for (const gdb_byte *q = p; q < aligned && q < end; q++)
	{
	  uint8_t expected = 0xf0 | (uint8_t) (aligned - q);
	  if (*q != expected)
	    {
	      pdb_dbg_printf ("LF_FIELDLIST: bad pad byte 0x%02x at offset %u "
			      "(expected 0x%02x)",
			      *q, (unsigned) (q - fl->data), expected);
	      break;
	    }
	}
      p = aligned;
      /* Need at least 2 bytes for the sub-record leaf type.  */
      if (p + 2 > end)
	break;

      uint16_t leaf = read_u16 (p);

      /* First compute the pointer to the next sub-record.  If there is a
	 problem we just bail out.  This makes it easier to check for the
	 correctness of the record before parsing it */
      const gdb_byte *next = pdb_fieldlist_skip_record (p, end, leaf);
      if (next == nullptr)
	{
	  pdb_warning ("LF_FIELDLIST: cannot skip sub-record 0x %04x, "
		       "remaining fields lost",
		       leaf);
	  break;
	}

      /* LF_INDEX (fieldlist continuation) applies to enums and compounds.  */
      if (leaf == LF_INDEX)
	{
	  /* Fieldlist continuation: a large fieldlist is split into multiple
	     LF_FIELDLIST records chained by LF_INDEX sub-records.  Switch to
	     the continuation record.  */
	  uint32_t cont_ti = read_u32 (p + LF_INDEX_TYPE_OFFS);
	  const pdb_tpi_type *cont = pdb_tpi_get_type (&pdb->tpi, cont_ti);
	  if (cont == nullptr || cont->leaf != LF_FIELDLIST)
	    {
	      pdb_warning ("LF_INDEX: continuation 0x%x invalid", cont_ti);
	      break;
	    }
	  if (std::find (visited.begin (), visited.end (), cont_ti)
	      != visited.end ())
	    {
	      pdb_warning ("LF_INDEX: continuation 0x%x forms a cycle",
			   cont_ti);
	      break;
	    }
	  visited.push_back (cont_ti);

	  p = cont->data;
	  end = cont->data + cont->data_len;
	  continue;
	}

      if (is_enum)
	{
	  /* Enum fieldlists carry only LF_ENUMERATE sub-records.  */
	  if (leaf == LF_ENUMERATE)
	    pdb_parse_lf_enumerate (pdb, p, end, fields);
	  else
	    pdb_dbg_printf ("LF_FIELDLIST(enum): unexpected sub-record "
			    "0x%04x",
			    leaf);
	}

      else
	{
	  /* Struct/class/union fieldlists.  */
	  switch (leaf)
	    {
	    case LF_MEMBER:
	      pdb_parse_lf_member (pdb, p, end, fields);
	      break;
	    case LF_BCLASS:
	      pdb_parse_lf_bclass (pdb, p, end, baseclasses);
	      break;
	    case LF_STMEMBER:
	      pdb_parse_lf_stmember (pdb, p, end, fields);
	      break;
	    case LF_NESTTYPE:
	      pdb_parse_lf_nesttype (pdb, p, end, nested_types);
	      break;
	    case LF_ONEMETHOD:
	      pdb_parse_lf_onemethod (pdb, p, end, fn_groups);
	      break;
	    case LF_METHOD:
	      pdb_parse_lf_method (pdb, p, end, fn_groups);
	      break;
	    case LF_VBCLASS:
	      pdb_parse_lf_vbclass (pdb, p, end, baseclasses);
	      break;
	    case LF_IVBCLASS:
	      /* Reached through a direct base; DWARF also records only direct
		 inheritance, so adding it would list a base the class of its
		 own does not declare.  */
	      break;
	    case LF_VFUNCTAB:
	      /* Remember the vtable pointer type.  */
	      if (!has_vfptr)
		{
		  uint32_t vftab_ti = read_u32 (p + LF_VFUNCTAB_TYPE_OFFS);
		  vfptr_type = &pdb_tpi_resolve_type (pdb, vftab_ti);
		  has_vfptr = true;
		}
	      break;
	    default:
	      pdb_dbg_printf ("LF_FIELDLIST: unknown sub-record 0x%04x at "
			      "offset %u",
			      leaf, (unsigned) (p - fl->data));
	      break;
	    }
	}

      p = next;
    }

  /* Give each override the vtable slot of the virtual it overrides,
     found in a base class.  */
  if (!baseclasses.empty ())
    for (auto &grp : fn_groups)
      for (auto &m : grp.methods)
	if (TYPE_FN_FIELD_VIRTUAL_P (&m, 0))
	  {
	    int vo = pdb_find_introducer_voffset (pdb, baseclasses, grp.name,
						  m.type);
	    if (vo >= 2)
	      m.voffset = vo;
	  }

  /* Cache parsed fieldlist data.  A fieldlist with methods or a vptr is
     unique to one class (a method's LF_MFUNCTION encodes its class type
     index), so it is never shared — don't waste a cache entry on it.  */
  if (fn_groups.empty () && !has_vfptr)
    pdb->tpi.fieldlist_cache.emplace (fieldlist_ti,
				      pdb_tpi_context::pdb_fieldlist_parse {
					baseclasses, fields, nested_types });

  result.baseclasses = std::move (baseclasses);
  result.fields = std::move (fields);
  result.nested_types = std::move (nested_types);
  result.fn_groups = std::move (fn_groups);
  result.has_vfptr = has_vfptr;
  result.vfptr_type = vfptr_type;
  return result;
}

/* Stitch a fieldlist onto TYPE:

   - Synthesise the vtable pointer.  CodeView marks the class that introduces
     the vtable with an LF_VFUNCTAB sub-record (captured by
     pdb_tpi_parse_fieldlist as HAS_VFPTR / VFPTR_TYPE) but emits no member for
     the pointer itself; GDB expects a synthetic "_vptr.Tag" data member.
     Insert it at bitpos 0 — the vptr lives at object offset 0 in the
     introducing class — and record its field index with
     set_type_vptr_fieldno.  The index is the base-class count, since bases
     occupy the leading slots and the vptr follows.
   - Qualify each static data member's physname as "Tag::name".
   - Qualify each non-friend method's physname as "Tag::name", adding a
     parameter suffix for overload groups when one is produced.  */

static void
pdb_apply_fieldlist (pdb_per_objfile *pdb, type *type,
		     const pdb_fieldlist_result &result)
{
  const std::vector<field> &baseclasses = result.baseclasses;
  const std::vector<decl_field> &nested_types = result.nested_types;
  const std::vector<pdb_fn_group> &fn_groups = result.fn_groups;
  bool has_vfptr = result.has_vfptr;
  struct type *vfptr_type = result.vfptr_type;

  const char *tag = type->name ();
  uint32_t nbaseclasses = baseclasses.size ();

  /* Synthesise "_vptr.Tag" at bitpos 0, right after the base classes.  */
  std::vector<field> fields = result.fields;
  if (has_vfptr && vfptr_type != nullptr)
    {
      const char *vname = obconcat (&pdb->objfile->objfile_obstack, "_vptr.",
				    tag != nullptr ? tag : "",
				    (char *) nullptr);
      field vf {};
      vf.set_name (vname);
      vf.set_type (vfptr_type);
      vf.set_loc_bitpos (0);
      vf.set_is_artificial (true);
      fields.insert (fields.begin (), vf);
    }

  uint32_t nfields = nbaseclasses + fields.size ();
  type->alloc_fields (nfields);

  uint32_t i = 0;
  for (const auto &f : baseclasses)
    type->field (i++) = f;
  for (const auto &f : fields)
    type->field (i++) = f;

  /* Qualify static data members as "Tag::name" (see pdb_parse_lf_stmember).  */
  if (tag != nullptr && *tag != '\0')
    for (uint32_t k = nbaseclasses; k < nfields; k++)
      {
	field &f = type->field (k);
	if (f.is_static ())
	  f.set_loc_physname (obconcat (&pdb->objfile->objfile_obstack, tag,
					"::", f.name (), (char *) nullptr));
      }

  if (nbaseclasses > 0 || !nested_types.empty () || !fn_groups.empty ()
      || has_vfptr)
    ALLOCATE_CPLUS_STRUCT_TYPE (type);

  if (nbaseclasses > 0)
    TYPE_N_BASECLASSES (type) = nbaseclasses;

  if (has_vfptr && vfptr_type != nullptr)
    {
      set_type_vptr_fieldno (type, (int) nbaseclasses);
      set_type_vptr_basetype (type, type);
    }

  if (!nested_types.empty ())
    {
      uint32_t count = nested_types.size ();
      decl_field *nested
	= (decl_field *) TYPE_ALLOC (type, count * sizeof (decl_field));
      for (uint32_t j = 0; j < count; j++)
	nested[j] = nested_types[j];

      TYPE_NESTED_TYPES_ARRAY (type) = nested;
      TYPE_NESTED_TYPES_COUNT (type) = count;
    }

  /* Attach method groups, qualifying each non-friend method as "Tag::name".
     A non-empty physname (the bare name from pdb_add_method_overload) means
     non-friend -> qualify it for this owner; an empty physname means friend
     (defined elsewhere as its own symbol) -> leave it untouched.  */
  if (!fn_groups.empty ())
    {
      auto ngroups = fn_groups.size ();
      fn_fieldlist *groups
	= (fn_fieldlist *) TYPE_ALLOC (type, ngroups * sizeof (fn_fieldlist));

      for (uint32_t j = 0; j < ngroups; j++)
	{
	  const pdb_fn_group &grp = fn_groups[j];
	  uint32_t nmethods = grp.methods.size ();
	  fn_field *methods
	    = (fn_field *) TYPE_ALLOC (type, nmethods * sizeof (fn_field));

	  /* gnuv3_pass_by_reference recognises a constructor by this flag or
	     by an Itanium-mangled physname, which MSVC never emits; without
	     it a class with a non-trivial copy ctor looks trivially copyable
	     and an inferior call returns it in a register instead of through
	     the hidden buffer.  A constructor's leaf name is the tag's.  */
	  bool group_is_ctor = false;
	  if (tag != nullptr && grp.name != nullptr)
	    {
	      const char *leaf = tag + pdb_last_component_offset (tag);
	      group_is_ctor = strcmp (grp.name, leaf) == 0;
	    }

	  for (uint32_t k = 0; k < nmethods; k++)
	    {
	      fn_field m = grp.methods[k];
	      m.is_constructor = group_is_ctor;
	      if (m.physname != nullptr && *m.physname != '\0')
		{
		  m.physname = pdb_build_method_physname (pdb, tag, grp.name);

		  /* The name alone reaches every overload, so carry the
		     signature that picks this one.  */
		  if (nmethods > 1)
		    {
		      std::string sig = pdb_method_param_signature (pdb, m);
		      if (!sig.empty ())
			m.physname
			  = obconcat (&pdb->objfile->objfile_obstack,
				      m.physname, sig.c_str (),
				      (char *) nullptr);
		    }
		}
	      methods[k] = m;
	    }

	  groups[j].name = grp.name;
	  groups[j].length = nmethods;
	  groups[j].fn_fields = methods;
	}

      /* Publish the pointer and count after every group and overload array
   is initialized.  Reentrant readers see no groups until this point;
   that does not imply construction of the whole type is complete.  */
      TYPE_FN_FIELDLISTS (type) = groups;
      TYPE_NFN_FIELDS (type) = static_cast<short> (ngroups);
    }

  /* MSVC returns a user-defined type in RAX only when it is POD-like: no
     user-defined constructor, destructor or copy assignment, no base class,
     no virtual function, and no private or protected non-static data member.
     Anything else comes back through the caller's buffer.  GDB's own
     triviality test models only the first two, and mingw's GCC returns the
     remaining shapes in RAX, so record the convention on this type instead of
     tightening the shared amd64-windows rule.  */
  if (type->code () == TYPE_CODE_STRUCT || type->code () == TYPE_CODE_UNION)
    {
      bool non_pod = nbaseclasses > 0 || has_vfptr;

      if (!non_pod && tag != nullptr)
	{
	  const char *leaf = tag + pdb_last_component_offset (tag);
	  for (const auto &grp : fn_groups)
	    if (grp.name != nullptr
		&& (grp.name[0] == '~'
		    || strcmp (grp.name, leaf) == 0
		    || strcmp (grp.name, "operator=") == 0))
	      {
		non_pod = true;
		break;
	      }
	}

      if (!non_pod)
	for (const auto &f : fields)
	  if (!f.is_static () && (f.is_private () || f.is_protected ()))
	    {
	      non_pod = true;
	      break;
	    }

      if (non_pod)
	{
	  ALLOCATE_CPLUS_STRUCT_TYPE (type);
	  TYPE_CPLUS_CALLING_CONVENTION (type) = DW_CC_pass_by_reference;
	}
    }
}

/* Read the field list FIELDLIST_TI and attach its members to the struct or
   union shell TYPE.  Types first referenced while reading get shells of
   their own, which are completed after this one.  */

static void
pdb_read_struct_members (pdb_per_objfile *pdb, type *type,
			 uint32_t fieldlist_ti)
{
  scoped_restore depth
    = make_scoped_restore (&pdb->tpi.struct_read_depth,
			   pdb->tpi.struct_read_depth + 1);

  pdb_fieldlist_result parse
    = pdb_tpi_parse_fieldlist (pdb, false, fieldlist_ti);
  pdb_apply_fieldlist (pdb, type, parse);
}

/* If the members of TYPE are still deferred, read them now.  */

static void
pdb_complete_struct (pdb_per_objfile *pdb, type *type)
{
  auto it = pdb->tpi.deferred_structs.find (type);
  if (it == pdb->tpi.deferred_structs.end ())
    return;

  uint32_t fieldlist_ti = it->second;
  pdb->tpi.deferred_structs.erase (it);
  pdb_read_struct_members (pdb, type, fieldlist_ti);
}

/* Read the members of every deferred shell, including shells deferred
   while this runs.  */

static void
pdb_complete_deferred_structs (pdb_per_objfile *pdb)
{
  pdb_tpi_context &tpi = pdb->tpi;

  for (size_t i = 0; i < tpi.deferred_struct_order.size (); i++)
    pdb_complete_struct (pdb, tpi.deferred_struct_order[i]);
  tpi.deferred_struct_order.clear ();
}

/* Type symbols.  */

/* Add one LOC_CONST symbol per enumerator of ENUM_TYPE to LIST, named in
  the enclosing scope: "ns::Color" gives "ns::Red".  Enumerators present
  only as LF_ENUMERATE fields need these symbols for lookup by name.
  The reader does not distinguish scoped from unscoped enums.  */

static void
pdb_add_enumerator_symbols (pdb_per_objfile *pdb, type *enum_type,
			    std::string_view name,
			    std::vector<struct symbol *> &list)
{
  if (enum_type->code () != TYPE_CODE_ENUM)
    return;

  std::string scope;
  std::string tag_name (name);
  size_t sep = pdb_last_component_offset (tag_name.c_str ());
  if (sep != 0)
    scope.assign (tag_name, 0, sep);

  for (int i = 0; i < enum_type->num_fields (); i++)
    {
      const field &f = enum_type->field (i);
      if (f.loc_kind () != FIELD_LOC_KIND_ENUMVAL || f.name () == nullptr)
	continue;

      std::string qname = scope + f.name ();
      auto *sym = new (&pdb->objfile->objfile_obstack) symbol;
      sym->set_language (language_cplus, &pdb->objfile->objfile_obstack);
      sym->compute_and_set_names (qname.c_str (), true,
				  pdb->objfile->per_bfd);
      sym->set_domain (VAR_DOMAIN);
      sym->set_loc_class_index (LOC_CONST);
      sym->set_type (enum_type);
      sym->set_value_longest (f.loc_enumval ());
      add_symbol_to_list (sym, list);
    }
}

/* Give each enclosing scope of NAME a namespace symbol in LIST.  A tagged
   type reached by name alone brings no symbol record with it, so without
   this the scope of a qualified tag has nothing to resolve against.

   TI's property word says what encloses the tag: a function body leaves no
   namespace at all, and a class supplies its own scopes rather than those
   spelled in NAME.  A scope naming a tagged type is a class, not a
   namespace.  */

static void
pdb_add_namespace_symbols (pdb_per_objfile *pdb, std::string_view name,
			   uint32_t ti, std::vector<symbol *> &list)
{
  uint16_t props = pdb_tpi_tag_props (&pdb->tpi, ti);

  if ((props & CV_PROP_SCOPED) != 0)
    return;

  if ((props & CV_PROP_ISNESTED) != 0)
    {
      auto owner = pdb->tpi.nested_owner.find (ti);
      if (owner == pdb->tpi.nested_owner.end ())
	return;

      const char *owner_name = pdb_tpi_tag_name (&pdb->tpi, owner->second);
      if (owner_name == nullptr || owner_name[0] == '\0')
	return;

      pdb_add_namespace_symbols (pdb, owner_name, owner->second, list);
      return;
    }

  std::string qname (name);

  pdb_for_each_scope_prefix (qname.c_str (), [&] (std::string_view scope)
    {
      std::string prefix (scope);

      if (pdb_tpi_is_tagged_type_name (&pdb->tpi, prefix.c_str ()))
	return;
      if (!pdb->built_namespace_names.insert (prefix).second)
	return;

      const char *ns_name = obstack_strdup (&pdb->objfile->objfile_obstack,
					    prefix);
      type *ns_type = type_allocator (pdb->objfile, language_cplus)
			.new_type (TYPE_CODE_NAMESPACE, 0, ns_name);

      symbol *ns_sym = pdb->objfile->new_symbol<symbol> ();
      ns_sym->set_language (language_cplus, &pdb->objfile->objfile_obstack);
      ns_sym->compute_and_set_names (ns_name, true, pdb->objfile->per_bfd);
      ns_sym->set_domain (TYPE_DOMAIN);
      ns_sym->set_loc_class_index (LOC_TYPEDEF);
      ns_sym->set_type (ns_type);
      add_symbol_to_list (ns_sym, list);
    });
}

/* See pdb-internal.h.  */

void
pdb_register_tpi_namespaces (pdb_per_objfile *pdb)
{
  bool any_qualified = false;
  for (const auto &entry : pdb->tpi.tagged_type_names)
    if (entry.first.find ("::") != std::string_view::npos)
      {
	any_qualified = true;
	break;
      }

  if (!any_qualified)
    return;

  scoped_restore decrementer = increment_reading_symtab ();

  buildsym_compunit cu (pdb->objfile, "<pdb-types>", "", language_cplus, 0);
  for (const auto &entry : pdb->tpi.tagged_type_names)
    pdb_add_namespace_symbols (pdb, entry.first, entry.second,
			       cu.get_global_symbols ());
  cu.end_compunit_symtab (0);
}

/* See pdb-internal.h.  */

void
pdb_register_tpi_typedefs (pdb_per_objfile *pdb)
{
  if (pdb->tpi.tagged_type_names.empty ())
    return;

  using clock = std::chrono::steady_clock;
  auto start = clock::now ();

  scoped_restore decrementer = increment_reading_symtab ();

  buildsym_compunit cu (pdb->objfile, "<pdb-types>", "", language_cplus, 0);
  uint32_t count = 0;
  for (const auto &entry : pdb->tpi.tagged_type_names)
    {
      std::string_view name = entry.first;
      uint32_t ti = entry.second;

      /* Skip forward references  */
      if (pdb_tpi_type_is_fwdref (&pdb->tpi, ti))
	continue;

      /* Skip types already built on demand (per-type lazy path);
	 mark the rest as built so the per-type path skips them.  */
      if (!pdb->built_type_names.insert (name).second)
	continue;

      type &gdb_type = pdb_tpi_resolve_type (pdb, ti);

      /* A global S_UDT already binds this name in <pdb-globals>; adding it
	 here too would list one type twice.  Its members still need the
	 symbols below.  */
      if (pdb->global_udt_names.count (name) == 0)
	{
	  auto *sym = new (&pdb->objfile->objfile_obstack) symbol;
	  sym->set_language (language_cplus, &pdb->objfile->objfile_obstack);
	  sym->compute_and_set_names (name.data (), true,
				      pdb->objfile->per_bfd);
	  sym->set_domain (STRUCT_DOMAIN);
	  sym->set_loc_class_index (LOC_TYPEDEF);
	  sym->set_type (&gdb_type);
	  add_symbol_to_list (sym, cu.get_global_symbols ());
	}
      pdb_add_namespace_symbols (pdb, name, ti, cu.get_global_symbols ());
      pdb_add_enumerator_symbols (pdb, &gdb_type, name,
				  cu.get_global_symbols ());
      count++;
      if (pdb_read_debug >= 1)
	debug_printf ("[pdb type-index] TYPE '%s' -> ti 0x%x\n",
		      name.data (), ti);
    }

  cu.end_compunit_symtab (0);

  if (pdb_read_debug >= 1)
    {
      double ms = std::chrono::duration<double, std::milli> (
			clock::now () - start).count ();
      debug_printf ("[pdb type-index] built <pdb-types>: %u tagged types"
		    " in %.2f ms\n", count, ms);
    }
}

/* See pdb-internal.h.  */

bool
pdb_build_tagged_type (pdb_per_objfile *pdb, std::string_view name)
{
  auto it = pdb->tpi.tagged_type_names.find (name);
  if (it == pdb->tpi.tagged_type_names.end ())
    return false;

  uint32_t ti = it->second;
  if (pdb_tpi_type_is_fwdref (&pdb->tpi, ti))
    return false;

  /* Build each tagged type once.  */
  if (!pdb->built_type_names.insert (it->first).second)
    return false;

  scoped_restore decrementer = increment_reading_symtab ();

  type &gdb_type = pdb_tpi_resolve_type (pdb, ti);

  buildsym_compunit cu (pdb->objfile, "<pdb-types>", "", language_cplus, 0);
  /* A global S_UDT already binds this name in <pdb-globals>; adding it here
     too would list one type twice.  Its members still need the symbols
     below.  */
  if (pdb->global_udt_names.count (it->first) == 0)
    {
      auto *sym = new (&pdb->objfile->objfile_obstack) symbol;
      sym->set_language (language_cplus, &pdb->objfile->objfile_obstack);
      sym->compute_and_set_names (it->first.data (), true,
				  pdb->objfile->per_bfd);
      sym->set_domain (STRUCT_DOMAIN);
      sym->set_loc_class_index (LOC_TYPEDEF);
      sym->set_type (&gdb_type);
      add_symbol_to_list (sym, cu.get_global_symbols ());
    }
  pdb_add_namespace_symbols (pdb, it->first, ti, cu.get_global_symbols ());
  pdb_add_enumerator_symbols (pdb, &gdb_type, it->first,
			      cu.get_global_symbols ());
  cu.end_compunit_symtab (0);

  if (pdb_read_debug >= 1)
    debug_printf ("[pdb type-index] built '%s' ti 0x%x (total %zu)\n",
		  it->first.data (), ti,
		  pdb->built_type_names.size ());
  return true;
}

/* See pdb-internal.h.  */

void
pdb_build_tagged_types_matching (pdb_per_objfile *pdb,
				 const lookup_name_info &lookup)
{
  lookup_name_info without_params = lookup.make_ignore_params ();
  symbol_name_matcher_ftype *matcher
    = language_def (language_cplus)->get_symbol_name_matcher (without_params);

  /* Collect first: building a type adds symbols and compunits, and the
     map must not be walked while that happens.  */
  std::vector<std::string_view> matches;

  for (const auto &entry : pdb->tpi.tagged_type_names)
    if (matcher (entry.first.data (), without_params, nullptr))
      matches.push_back (entry.first);

  for (std::string_view name : matches)
    pdb_build_tagged_type (pdb, name);
}

/* See pdb-internal.h.  */

bool
pdb_build_enum_for_enumerator (pdb_per_objfile *pdb,
			       const lookup_name_info &lookup)
{
  if (pdb->tpi.enumerator_names.empty ())
    return false;

  lookup_name_info without_params = lookup.make_ignore_params ();
  symbol_name_matcher_ftype *matcher
    = language_def (language_cplus)->get_symbol_name_matcher (without_params);

  /* Collect first: building a type adds symbols and compunits.  */
  std::vector<const char *> tags;
  auto consider = [&] (std::string_view enumerator, const char *tag)
    {
      /* Named as pdb_add_enumerator_symbols names the symbol.  */
      std::string qname (tag, pdb_last_component_offset (tag));
      qname.append (enumerator);
      if (matcher (qname.c_str (), without_params, nullptr))
	tags.push_back (tag);
    };

  if (lookup.completion_mode ())
    for (const auto &[enumerator, tag] : pdb->tpi.enumerator_names)
      consider (enumerator, tag);
  else
    {
      const char *name = without_params.c_str ();
      auto range = pdb->tpi.enumerator_names.equal_range
	(std::string_view (name + pdb_last_component_offset (name)));
      for (auto it = range.first; it != range.second; ++it)
	consider (it->first, it->second);
    }

  bool built = false;
  for (const char *tag : tags)
    built |= pdb_build_tagged_type (pdb, std::string_view (tag));
  return built;
}

/* namespace pdb */
}
