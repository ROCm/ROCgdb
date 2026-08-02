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

/* This file implements TPI (Type Program Information) stream parsing
   and type resolution for use with GDB symbols.

   References:
     - microsoft-pdb: https://github.com/microsoft/microsoft-pdb
     - LLVM docs: https://llvm.org/docs/PDB/TpiStream.html

   CURRENT LIMITATIONS:
   Not all types are supported (e.g. LF_INTERFACE, templates).
   Unsupported types are mapped to TYPE_CODE_ERROR and display as
   "<unsupported PDB type>".
*/

#include "symtab.h"
#include "gdbtypes.h"
#include "objfiles.h"
#include "buildsym.h"
#include "complaints.h"
#include "pdb/pdb-internal.h"
#include <string.h>

namespace pdb
{

/* ---------------------------------------------------------------
   TPI / IPI Stream (Type Program Information)
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
	  RecordData (RecordLen - 2 bytes) — fields depend on RecordKind.
   ---------------------------------------------------------------.  */

/* TPI Stream Header defines used during parsing.  */
inline constexpr auto TPI_HDR_VERSION_OFFS = 0 /* Version.  */;
inline constexpr auto TPI_HDR_HEADER_SIZE_OFFS = 4 /* HeaderSize.  */;
inline constexpr auto TPI_HDR_TYPE_INDEX_BEGIN_OFFS = 8 /* TypeIndexBegin.  */;
inline constexpr auto TPI_HDR_TYPE_INDEX_END_OFFS = 12 /* TypeIndexEnd.  */;
inline constexpr auto TPI_HDR_TYPE_REC_BYTES_OFFS = 16 /* TypeRecordBytes.  */;
inline constexpr auto TPI_HDR_SIZE = 56 /* Total header size.  */;

inline constexpr auto TPI_VERSION_V80 = 20040203 /* Only observed version.  */;

struct pdb_lf_pointer_attr
{
  unsigned int ptrtype
    : 5; /* Ordinal specifying pointer type (CV_ptrtype_e).  */
  unsigned int ptrmode
    : 3; /* Ordinal specifying pointer mode (CV_ptrmode_e).  */
  unsigned int isflat32 : 1;    /* True if 0:32 pointer.  */
  unsigned int isvolatile : 1;  /* TRUE if volatile pointer.  */
  unsigned int isconst : 1;     /* TRUE if const pointer.  */
  unsigned int isunaligned : 1; /* TRUE if unaligned pointer.  */
  unsigned int isrestrict : 1;  /* TRUE if restricted pointer.  */
  unsigned int size : 6;        /* Size of pointer (in bytes).  */
  unsigned int ismocom : 1;     /* TRUE if MoCOM pointer (^ or %).  */
  unsigned int islref : 1;      /* TRUE if & ref-qualifier.  */
  unsigned int isrref : 1;      /* TRUE if && ref-qualifier.  */
  unsigned int unused : 10;     /* Pad out to 32-bits.  */
};

struct pdb_lf_pointer
{
  uint32_t utype;           /* Type index of underlying type.  */
  pdb_lf_pointer_attr attr; /* Attributes bitfield.  */
};

/* LF_ARRAY sub-record layout.  */

struct pdb_lf_array
{
  uint32_t elemtype; /* Type index of element type.  */
  uint32_t idxtype;  /* Type index of indexing type.  */
  unsigned char
    data[]; /* Variable-length: numeric leaf for size, then name.  */
};

/* Forward declarations.  */
static type *pdb_tpi_resolve_type_internal (pdb_per_objfile *pdb,
					    uint32_t type_idx);
static const pdb_tpi_type *pdb_tpi_get_type (const pdb_tpi_context *tpi,
					     uint32_t type_idx);
uint32_t pdb_cv_read_numeric (const gdb_byte *data, uint32_t max_len,
			      uint64_t *value);
static type *pdb_tpi_get_unsupported_type (pdb_per_objfile *pdb);

/* Get a type record from the TPI context by type index.
   Returns nullptr if the index is out of range.  */
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

/* Parse raw stream data into a pdb_tpi_context.
   RECORDS points to the start of the type records (after header),
   rec_bytes is the size of the record area.  */
static bool
pdb_parse_tpi_records (pdb_per_objfile *pdb, const gdb_byte *records,
		       uint32_t rec_bytes, const char *stream_name,
		       pdb_tpi_context &tpi)
{
  uint32_t num_types = tpi.type_idx_end - tpi.type_idx_begin;

  /* Allocate type record array on obstack - will be accessed
     during any module's expansion, type by type...  */
  pdb_tpi_type *types = nullptr;
  if (num_types > 0)
    types = OBSTACK_CALLOC (&pdb->objfile->objfile_obstack, num_types,
			    pdb_tpi_type);

  /* Walk the records and parse into tpi->types. */
  const gdb_byte *p = records;
  const gdb_byte *rec_end = p + rec_bytes;
  uint32_t idx = 0;

  while (idx < num_types)
    {
      if (p + CV_REC_HDR_SIZE > rec_end)
	{
	  pdb_error ("%s: unexpected end of records at type 0x%x", stream_name,
		     tpi.type_idx_begin + idx);
	}

      auto rec_len = read_u16 (p);
      auto rec_kind = read_u16 (p + 2);

      if (rec_len < 2)
	{
	  pdb_error ("%s: type 0x%x has invalid RecordLen %u", stream_name,
		     tpi.type_idx_begin + idx, rec_len);
	}

      /* rec_len includes the 2-byte RecordKind but not the 2-byte RecordLen
	 field itself.  Total on-disk size = 2 (RecordLen) + rec_len.  */
      auto rec_size = rec_len + 2;

      if (p + rec_size > rec_end)
	{
	  pdb_error ("%s: type 0x%x overflows record area", stream_name,
		     tpi.type_idx_begin + idx);
	}

      types[idx].leaf = rec_kind;
      types[idx].length = rec_len;
      types[idx].data = p + CV_REC_HDR_SIZE;
      types[idx].data_len = rec_len - 2;

      idx++;

      /* Advance to next record (4-byte aligned). */
      p += rec_size;
      p = (const gdb_byte *) align_up ((uintptr_t) p, 4);
    }

  tpi.types = types;
  return true;
}

/* Read and parse TPI (stream 2) if present.
   Populates pdb->tpi->types[] with decoded type records.  */
bool
pdb_read_tpi_stream (pdb_per_objfile *pdb)
{
  auto &tpi = pdb->tpi;

  if (pdb->num_streams <= PDB_STREAM_TPI
      || pdb->stream_sizes[PDB_STREAM_TPI] == 0)
    {
      pdb_warning ("TPI stream (index 2) missing or empty");
      return false;
    }

  /* Cache TPI stream — type records are referenced on demand.  */
  auto tpi_buf = pdb_read_stream (pdb, PDB_STREAM_TPI);
  if (tpi_buf == nullptr)
    {
      pdb_error ("Failed to read TPI stream");
    }

  auto *tpi_data = tpi_buf.release ();
  pdb->stream_data[PDB_STREAM_TPI] = tpi_data;

  pdb->tpi_size = pdb->stream_sizes[PDB_STREAM_TPI];

  if (pdb->tpi_size < TPI_HDR_SIZE)
    {
      pdb_error ("TPI stream too small (%u bytes)", pdb->tpi_size);
    }

  uint32_t version = read_u32 (tpi_data + TPI_HDR_VERSION_OFFS);
  uint32_t hdr_size = read_u32 (tpi_data + TPI_HDR_HEADER_SIZE_OFFS);
  uint32_t rec_bytes = read_u32 (tpi_data + TPI_HDR_TYPE_REC_BYTES_OFFS);
  tpi.type_idx_begin = read_u32 (tpi_data + TPI_HDR_TYPE_INDEX_BEGIN_OFFS);
  tpi.type_idx_end = read_u32 (tpi_data + TPI_HDR_TYPE_INDEX_END_OFFS);

  if (version != TPI_VERSION_V80)
    {
      pdb_error ("TPI unknown version 0x%08x (expected 0x%08x)", version,
		 TPI_VERSION_V80);
    }

  if (hdr_size < TPI_HDR_SIZE)
    {
      pdb_error ("TPI header size %u too small", hdr_size);
    }

  if (hdr_size + rec_bytes > pdb->tpi_size)
    {
      pdb_error ("TPI records overflow stream (%u + %u > %u)", hdr_size,
		 rec_bytes, pdb->tpi_size);
    }

  /* Log TPI stream info before parsing.  */
  uint32_t num_types = tpi.type_idx_end - tpi.type_idx_begin;
  pdb_dbg_printf ("TPI: TI=[0x%x..0x%x)  num_types=%u  rec_bytes=%u",
		  tpi.type_idx_begin, tpi.type_idx_end, num_types, rec_bytes);

  if (!pdb_parse_tpi_records (pdb, tpi_data + hdr_size, rec_bytes, "TPI", tpi))
    {
      return false;
    }

  /* Allocate unified type cache covering simple types (0x0000-0x0FFF)
     and all compound types up to type_idx_end.  */
  uint32_t cache_size = tpi.type_idx_end > 0x1000 ? tpi.type_idx_end : 0x1000;
  pdb->tpi.type_cache = OBSTACK_CALLOC (&pdb->objfile->objfile_obstack,
					cache_size, type *);

  return true;
}

/* Resolve a simple (built-in) type index to a GDB type.
 Simple type indices are < 0x1000 and encode:
     bits 0-7  : SimpleTypeKind
     bits 8-11 : SimpleTypeMode (direct, near pointer, etc.).
   Results are cached in pdb_tpi_context to avoid recreating
   custom-sized pointers.  */
static type *
pdb_tpi_resolve_simple_type (pdb_per_objfile *pdb, uint32_t type_idx)
{
  /* Check cache first.  */
  if (pdb->tpi.type_cache != nullptr)
    {
      type *cached = pdb->tpi.type_cache[type_idx];
      if (cached != nullptr)
	return cached;
    }

  gdbarch *garch = pdb->objfile->arch ();
  const struct builtin_type *bt = builtin_type (garch);

  uint32_t kind = CV_SIMPLE_KIND (type_idx);
  uint32_t mode = CV_SIMPLE_MODE (type_idx);

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
      base = bt->builtin_char16;
      break;

    case CV_CHAR16:
      base = bt->builtin_char16;
      break;

    case CV_CHAR32:
      base = bt->builtin_char32;
      break;

    case CV_SBYTE:
      base = bt->builtin_int8;
      break;

    case CV_BYTE:
      base = bt->builtin_uint8;
      break;

    case CV_INT16:
      base = bt->builtin_int16;
      break;

    case CV_UINT16:
      base = bt->builtin_uint16;
      break;

    case CV_INT32:
      base = bt->builtin_int32;
      break;

    case CV_UINT32:
      base = bt->builtin_uint32;
      break;

    case CV_LONG:
      base = bt->builtin_int32;
      break;

    case CV_ULONG:
      base = bt->builtin_uint32;
      break;

    case CV_QUAD:
    case CV_INT64:
      base = bt->builtin_int64;
      break;

    case CV_UQUAD:
    case CV_UINT64:
      base = bt->builtin_uint64;
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

    case CV_BOOL8:
      base = bt->builtin_bool;
      break;

    case CV_HRESULT:
      base = bt->builtin_int32;
      break;

    default:
      pdb_warning ("Unknown simple type 0x%02x in TI 0x%04x", kind, type_idx);
      return pdb_tpi_get_unsupported_type (pdb);
    }

  if (mode == CV_TM_DIRECT)
    return base;

  /* Wrap in a pointer. First determine pointer size from mode.  */
  int ptr_size;
  switch (mode)
    {
      /* See CV_prmode_e - only some pointers have specified size.  */
    case CV_TM_NPTR32:
    case CV_TM_FPTR32:
      ptr_size = 4;
      break;
    case CV_TM_NPTR64:
      ptr_size = 8;
      break;
    case CV_TM_NPTR128:
      ptr_size = 16;
      break;
    default:
      /* legacy modes (NPTR, FPTR, HPTR) — use arch default.  */
      ptr_size = gdbarch_ptr_bit (garch) / 8;
      break;
    }

  /* Skip lookup if size doesn't match arch default.  */
  int arch_ptr_size = gdbarch_ptr_bit (garch) / 8;
  type *ptr_type;

  if (ptr_size == arch_ptr_size)
    {
      /* Standard arch size, use GDB's lookup cache.  */
      ptr_type = lookup_pointer_type (base);
    }

  else
    {
      /* Non-standard size, create directly.  */
      type_allocator alloc (pdb->objfile->arch ());
      ptr_type = init_pointer_type (alloc, ptr_size * 8, nullptr, base);
    }

  /* Cache the result before returning.  */
  if (pdb->tpi.type_cache != nullptr)
    pdb->tpi.type_cache[type_idx] = ptr_type;
  return ptr_type;
}

/* Create a custom UNDEFINED type for unsupported types.
   Cached per-objfile in pdb_tpi_context::undefined_type.  */
static type *
pdb_tpi_get_unsupported_type (pdb_per_objfile *pdb)
{
  if (pdb->tpi.undefined_type != nullptr)
    return pdb->tpi.undefined_type;

  /* Create an error type for unsupported/unresolved PDB types.
     TYPE_CODE_ERROR tells GDB's value-printing infrastructure to print
     "<error type>" instead of trying to interpret the value as a struct,
     which would crash.  */
  type_allocator alloc (pdb->objfile->arch ());
  type *undef = alloc.new_type (TYPE_CODE_ERROR, 8, nullptr);
  undef->set_name ("<unsupported PDB type>");

  pdb->tpi.undefined_type = undef;
  return undef;
}

/* Read a CodeView numeric leaf — a variable-length encoded integer used
   for array sizes, dimensions, and other numeric fields in type records.

   Returns: Number of bytes consumed (0 if error or invalid).
   Output:  *value = the decoded numeric value.

   We first read the first uint16. If it's < 0x8000, the uint16 itself is the
   value (and 2 bytes are consumed). If it's >= 0x8000, the uint16 is a tag
   that says what type of integer follows (LF_CHAR = 1 byte, LF_SHORT = 2 bytes,
   LF_LONG = 4 bytes, etc.).

   Caller must use the return value to advance the data pointer to skip
   past this leaf if parsing sequential fields.  */
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

/* LF_MODIFIER  */
static type *
pdb_tpi_make_modifier (pdb_per_objfile *pdb, const pdb_tpi_type *rec)
{
  if (!pdb_check_record_size (rec, LF_MOD_SIZE, "LF_MODIFIER"))
    return pdb_tpi_get_unsupported_type (pdb);

  auto type_idx = read_u32 (rec->data + LF_MOD_TYPE_OFFS);
  auto attr = read_u16 (rec->data + LF_MOD_ATTR_OFFS);

  auto *base = pdb_tpi_resolve_type_internal (pdb, type_idx);

  bool is_const = (attr & CV_MODIFIER_CONST) != 0;
  bool is_volatile = (attr & CV_MODIFIER_VOLATILE) != 0;
  if (is_const || is_volatile)
    base = make_cv_type (is_const, is_volatile, base);

  return base;
}

/* LF_POINTER  */
static type *
pdb_tpi_make_pointer (pdb_per_objfile *pdb, const pdb_tpi_type *rec)
{
  if (!pdb_check_record_size (rec, sizeof (pdb_lf_pointer), "LF_POINTER"))
    return pdb_tpi_get_unsupported_type (pdb);

  auto ptr = (const pdb_lf_pointer *) rec->data;

  auto *pointee = pdb_tpi_resolve_type_internal (pdb, ptr->utype);

  type *result;
  switch (ptr->attr.ptrmode)
    {
    case CV_PTRMODE_LVALUE_REF:
      result = lookup_lvalue_reference_type (pointee);
      break;
    case CV_PTRMODE_POINTER:
      result = lookup_pointer_type (pointee);
      break;
    case CV_PTRMODE_RVALUE_REF:
      result = lookup_rvalue_reference_type (pointee);
      break;
    default:
      result = lookup_pointer_type (pointee);
      break;
    }

  if (ptr->attr.isconst || ptr->attr.isvolatile)
    result = make_cv_type (ptr->attr.isconst, ptr->attr.isvolatile, result);

  /* TODO: validate or enforce pointer size (ptr->attr.size) if needed.  */

  return result;
}

/* LF_ARRAY  */
static type *
pdb_tpi_make_array (pdb_per_objfile *pdb, const pdb_tpi_type *rec)
{
  if (!pdb_check_record_size (rec, sizeof (pdb_lf_array), "LF_ARRAY"))
    return pdb_tpi_get_unsupported_type (pdb);

  auto array = (const pdb_lf_array *) rec->data;

  auto *elem_type = pdb_tpi_resolve_type_internal (pdb, array->elemtype);
  auto *idx_type = pdb_tpi_resolve_type_internal (pdb, array->idxtype);

  /* Read total array size from the data[] field.  */
  uint64_t total_size = 0;
  uint64_t consumed = 0;
  uint32_t data_len = rec->data_len > 8 ? (rec->data_len - 8) : 0;
  if (data_len > 0)
    {
      consumed = pdb_cv_read_numeric (array->data, data_len, &total_size);
    }

  if (consumed == 0 && data_len > 0)
    pdb_dbg_printf ("LF_ARRAY: failed to read numeric leaf for array size");

  /* TODO: What about the name after consumed bytes? */

  uint64_t elem_size = elem_type->length ();
  uint64_t num_elements = 0;
  if (elem_size > 0 && total_size > 0)
    num_elements = total_size / elem_size;

  type_allocator alloc (pdb->objfile->arch ());
  auto *range_type
    = create_static_range_type (alloc, idx_type, 0,
				num_elements > 0 ? num_elements - 1 : 0);
  auto *array_type = create_array_type (alloc, elem_type, range_type);
  array_type->set_length (total_size);
  return array_type;
}

/* LF_BITFIELD.  */
static type *
pdb_tpi_make_bitfield (pdb_per_objfile *pdb, const pdb_tpi_type *rec)
{
  if (!pdb_check_record_size (rec, LF_BITFIELD_SIZE, "LF_BITFIELD"))
    return pdb_tpi_get_unsupported_type (pdb);

  auto base_ti = read_u32 (rec->data + LF_BITFIELD_TYPE_OFFS);

  return pdb_tpi_resolve_type_internal (pdb, base_ti);
}

/* Resolve argument list from LF_ARGLIST record and populate func_type fields.
   EXTRA_SLOTS reserves leading field slots (e.g. 1 for an implicit 'this'
   parameter that the caller fills separately).  Arglist entries are placed
   starting at field index EXTRA_SLOTS.  */
static void
pdb_tpi_resolve_arglist (pdb_per_objfile *pdb, type *func_type,
			 uint32_t arglist_ti, uint32_t extra_slots = 0)
{
  auto *rec = pdb_tpi_get_type (&pdb->tpi, arglist_ti);
  if (rec == nullptr)
    {
      pdb_warning ("LF_ARGLIST: type index 0x%x out of range", arglist_ti);
      return;
    }

  if (rec->leaf != LF_ARGLIST)
    {
      pdb_warning ("LF_ARGLIST: type index 0x%x has unexpected leaf 0x%04x",
		   arglist_ti, rec->leaf);
      return;
    }

  if (!pdb_check_record_size (rec, LF_ARGLIST_MIN_SIZE, "LF_ARGLIST"))
    return;

  auto argc = read_u32 (rec->data + LF_ARGLIST_COUNT_OFFS);
  if (rec->data_len < LF_ARGLIST_ARGS_OFFS + argc * 4)
    return;

  uint32_t total = extra_slots + argc;
  if (total == 0)
    return;

  func_type->alloc_fields (total);
  for (uint32_t i = 0; i < argc; i++)
    {
      auto arg_ti = read_u32 (rec->data + LF_ARGLIST_ARGS_OFFS + i * 4);
      auto *arg_type = pdb_tpi_resolve_type_internal (pdb, arg_ti);
      func_type->field (extra_slots + i).set_type (arg_type);
    }
}

/* LF_PROCEDURE  */
static type *
pdb_tpi_make_procedure (pdb_per_objfile *pdb, const pdb_tpi_type *rec)
{
  if (!pdb_check_record_size (rec, LF_PROC_SIZE, "LF_PROCEDURE"))
    return pdb_tpi_get_unsupported_type (pdb);

  auto ret_ti = read_u32 (rec->data + LF_PROC_RVTYPE_OFFS);
  auto arglist_ti = read_u32 (rec->data + LF_PROC_ARGLIST_OFFS);

  auto *ret_type = pdb_tpi_resolve_type_internal (pdb, ret_ti);
  auto *func_type = lookup_function_type (ret_type);

  pdb_tpi_resolve_arglist (pdb, func_type, arglist_ti);
  return func_type;
}

/* LF_MFUNCTION — Member function.
   Reads class_ti, this_ti, ret_ti and arglist_ti from the record.
   Builds a TYPE_CODE_METHOD with 'this' as artificial fields[0]
   and self_type set to the containing class, matching DWARF's
   read_subroutine_type + dwarf2_add_member_fn + smash_to_method_type
   flow.  Static methods (this_ti == 0) get no 'this' parameter.  */
static type *
pdb_tpi_make_mfunction (pdb_per_objfile *pdb, const pdb_tpi_type *rec)
{
  if (!pdb_check_record_size (rec, LF_MFUNC_SIZE, "LF_MFUNCTION"))
    return pdb_tpi_get_unsupported_type (pdb);

  auto ret_ti = read_u32 (rec->data + LF_MFUNC_RVTYPE_OFFS);
  auto class_ti = read_u32 (rec->data + LF_MFUNC_CLASSTYPE_OFFS);
  auto this_ti = read_u32 (rec->data + LF_MFUNC_THISTYPE_OFFS);
  auto arglist_ti = read_u32 (rec->data + LF_MFUNC_ARGLIST_OFFS);

  auto *ret_type = pdb_tpi_resolve_type_internal (pdb, ret_ti);
  auto *func_type = lookup_function_type (ret_type);

  /* For non-static methods (this_ti != 0), reserve slot 0 for the
     implicit 'this' pointer and fill arglist starting at slot 1.
     This avoids a second allocation + copy.  DWARF does the same:
     read_subroutine_type puts 'this' in fields[0] with
     DW_AT_artificial, and cp_type_print_method_args skips it.  */
  bool has_this = (this_ti != 0);
  pdb_tpi_resolve_arglist (pdb, func_type, arglist_ti, has_this ? 1 : 0);

  if (has_this)
    {
      auto *this_type = pdb_tpi_resolve_type_internal (pdb, this_ti);
      func_type->field (0).set_type (this_type);
      func_type->field (0).set_is_artificial (true);
    }

  /* Convert TYPE_CODE_FUNC → TYPE_CODE_METHOD and set self_type to
     the containing class.  smash_to_method_type zeros the type then
     re-assigns the fields pointer (obstack memory survives).  */
  auto *class_type = pdb_tpi_resolve_type_internal (pdb, class_ti);
  gdb::array_view<field> args = func_type->fields ();
  smash_to_method_type (func_type, class_type, ret_type, args, 0);

  return func_type;
}

/* ---------------------------------------------------------------
   Compound types: LF_STRUCTURE, LF_CLASS, LF_UNION, LF_ENUM.
   --------------------------------------------------------------- */

/* Set field accessibility from CV_fldattr_t access bits.
   Mirrors DWARF's DW_AT_accessibility handling.  */
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
      /* Public is default, no need to set.  */
      break;
    default:
      break;
    }
}

/* Advance P past padding bytes that separate sub-records in an
   LF_FIELDLIST.  Sub-records are variable-length, so MSVC inserts
   1-3 padding bytes after each one to align the next sub-record to
   a 4-byte boundary.  Each padding byte has a value in 0xf0..0xff;
   the low nibble encodes the total number of padding bytes
   (including itself): 0xf1 = 1 byte, 0xf2 = 2, 0xf3 = 3.  */
static const gdb_byte *
pdb_fieldlist_skip_padding (const gdb_byte *p, const gdb_byte *end)
{
  while (p < end)
    {
      uint8_t b = *p;
      if (b >= 0xf0 && b < LF_NUMERIC)
	p += b & 0x0f;
      else
	break;
    }

  return p;
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
inline constexpr auto CV_MPROP_INTRO = 4 /* Introducing virtual.  */;
inline constexpr auto CV_MPROP_PUREVIRT = 5;
inline constexpr auto CV_MPROP_PUREINTRO = 6 /* Pure introducing virtual.  */;

/* Extract CV_methodprop_e from CV_fldattr_t.  */
#define CV_MPROP(attr) (((attr) >> 2) & 0x07)

/* True when mprop is INTRO or PUREINTRO (extra uint32_t vbaseoff follows).  */
#define CV_MPROP_HAS_VBASEOFF(mprop) \
  ((mprop) == CV_MPROP_INTRO || (mprop) == CV_MPROP_PUREINTRO)

/* Advance past a single sub-record at P, returning pointer past it.
   Returns nullptr if the record is truncated.

   Sub-records have three variable-length patterns:
     1. fixed fields + numeric leaf + NUL name  (LF_MEMBER, LF_ENUMERATE)
     2. fixed fields + numeric leaf (no name)   (LF_BCLASS)
     3. fixed fields + NUL name (no numeric)    (LF_STMEMBER, LF_NESTTYPE,
					 LF_METHOD)  */
static const gdb_byte *
pdb_fieldlist_skip_record (const gdb_byte *p, const gdb_byte *end,
			   uint16_t leaf)
{
  /* Skip past numeric leaf + NUL-terminated name starting at DATA_OFFS.  */
  auto skip_numeric_name = [&] (uint32_t data_offs) -> const gdb_byte *
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
  auto skip_numeric = [&] (uint32_t data_offs) -> const gdb_byte *
    {
      if (p + data_offs > end)
	return nullptr;

      const gdb_byte *d = p + data_offs;
      uint64_t dummy;
      uint32_t nr = pdb_cv_read_numeric (d, (uint32_t) (end - d), &dummy);
      return (nr == 0) ? nullptr : d + nr;
    };

  /* Skip past NUL-terminated name starting at NAME_OFFS.  */
  auto skip_name = [&] (uint32_t name_offs) -> const gdb_byte *
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
	  d += 4; /* skip vbaseoff  */
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

/* Group of overloaded methods with the same name. Used to collect all
   overloads in an LF_METHODLIST under a single name entry.  */
struct pdb_fn_group
{
  const char *name;
  std::vector<fn_field> methods;
};

/* Search for an existing method group by NAME. If found, return pointer to it.
   If not found, create a new group with that name and return it.
   Used when parsing LF_ONEMETHOD and LF_METHOD sub-records: multiple
   overloads of the same function name are collected into one group.  */
static pdb_fn_group *
pdb_find_or_add_fn_group (std::vector<pdb_fn_group> &groups, const char *name)
{
  for (auto &g : groups)
    if (strcmp (g.name, name) == 0)
      return &g;

  groups.emplace_back ();
  groups.back ().name = name;
  return &groups.back ();
}

/* Populate a single fn_field from method attributes and type index.  */
static void
pdb_fill_fn_field (pdb_per_objfile *pdb, fn_field *fnp, uint16_t attr,
		   uint32_t type_ti)
{
  memset (fnp, 0, sizeof (*fnp));
  fnp->type = pdb_tpi_resolve_type_internal (pdb, type_ti);
  fnp->physname = "";

  uint16_t access = attr & CV_ACCESS_MASK;
  switch (access)
    {
    case CV_ACCESS_PRIVATE:
      fnp->accessibility = accessibility::PRIVATE;
      break;

    case CV_ACCESS_PROTECTED:
      fnp->accessibility = accessibility::PROTECTED;
      break;

    default:
      fnp->accessibility = accessibility::PUBLIC;
      break;
    }

  uint16_t mprop = CV_MPROP (attr);
  switch (mprop)
    {
    case CV_MPROP_STATIC:
      fnp->voffset = VOFFSET_STATIC;
      break;
    case CV_MPROP_VIRTUAL:
    case CV_MPROP_INTRO:
    case CV_MPROP_PUREVIRT:
    case CV_MPROP_PUREINTRO:
      /* For now, mark as virtual with a placeholder offset.
	 We'd need the vbaseoff to compute the real slot.  */
      fnp->voffset = 2; /* > 1 means virtual.  */
      break;
    default:
      fnp->voffset = 0;
      break;
    }
}

/* ---------------------------------------------------------------
   LF_FIELDLIST sub-record handlers.

   Each pdb_parse_lf_* function handles one sub-record leaf type from
   an LF_FIELDLIST.  They read fields from the raw sub-record at P,
   resolve any type indices, and push the result into the appropriate
   output vector (fields, baseclasses, nested_types, or fn_groups).

   --------------------------------------------------------------- */

/* LF_MEMBER — non-static data member of a struct/union.  */
static void
pdb_parse_lf_member (pdb_per_objfile *pdb, const gdb_byte *p,
		     const gdb_byte *end, std::vector<field> &fields)
{
  uint16_t attr = read_u16 (p + LF_MEMBER_ATTR_OFFS);
  uint32_t type_ti = read_u32 (p + LF_MEMBER_TYPE_OFFS);
  const gdb_byte *d = p + LF_MEMBER_DATA_OFFS;

  /* Pointer to the actual member data. */
  uint64_t offset_val;
  auto nr = pdb_cv_read_numeric (d, (uint32_t) (end - d), &offset_val);
  d += nr;
  auto name = CSTR (d);

  type *ftype = pdb_tpi_resolve_type_internal (pdb, type_ti);
  field f;
  memset (&f, 0, sizeof (f));
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
  d += nr;
  auto name = CSTR (d);

  field f;
  memset (&f, 0, sizeof (f));
  f.set_name (name);
  f.set_loc_enumval ((LONGEST) val);
  pdb_set_field_accessibility (&f, attr);
  fields.push_back (f);
}

/* LF_BCLASS — direct (non-virtual) base class.
   Field layout defined by LF_BCLASS_* offsets in pdb-internal.h.  */
static void
pdb_parse_lf_bclass (pdb_per_objfile *pdb, const gdb_byte *p,
		     const gdb_byte *end, std::vector<field> &baseclasses)
{
  uint16_t attr = read_u16 (p + LF_BCLASS_ATTR_OFFS);
  uint32_t base_ti = read_u32 (p + LF_BCLASS_TYPE_OFFS);
  const gdb_byte *d = p + LF_BCLASS_DATA_OFFS;

  uint64_t offset;
  pdb_cv_read_numeric (d, (uint32_t) (end - d), &offset);

  type *base_type = pdb_tpi_resolve_type_internal (pdb, base_ti);
  field f;
  memset (&f, 0, sizeof (f));
  f.set_type (base_type);
  f.set_name (base_type->name () ? base_type->name () : "");
  f.set_loc_bitpos (offset * 8);
  pdb_set_field_accessibility (&f, attr);
  baseclasses.push_back (f);
}

/* LF_STMEMBER — static data member.
   Linkage name not available - leave empty physname (which GDB uses to
   recognize this as static).  */
static void
pdb_parse_lf_stmember (pdb_per_objfile *pdb, const gdb_byte *p,
		       const gdb_byte *end, std::vector<field> &fields)
{
  uint16_t attr = read_u16 (p + LF_STMEMBER_ATTR_OFFS);
  uint32_t type_ti = read_u32 (p + LF_STMEMBER_TYPE_OFFS);
  auto name = CSTR (p + LF_STMEMBER_NAME_OFFS);

  type *ftype = pdb_tpi_resolve_type_internal (pdb, type_ti);
  field f;
  memset (&f, 0, sizeof (f));
  f.set_name (name);
  f.set_type (ftype);
  f.set_loc_physname ("");
  pdb_set_field_accessibility (&f, attr);
  fields.push_back (f);
}

/* LF_NESTTYPE — nested type definition (typedef inside a class).  */
static void
pdb_parse_lf_nesttype (pdb_per_objfile *pdb, const gdb_byte *p,
		       const gdb_byte *end [[maybe_unused]],
		       std::vector<decl_field> &nested_types)
{
  uint32_t nested_ti = read_u32 (p + LF_NESTTYPE_TYPE_OFFS);
  auto name = CSTR (p + LF_NESTTYPE_NAME_OFFS);

  type *nested_type = pdb_tpi_resolve_type_internal (pdb, nested_ti);
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
  if (CV_MPROP_HAS_VBASEOFF (CV_MPROP (attr)))
    d += 4;
  auto name = CSTR (d);

  pdb_fn_group *grp = pdb_find_or_add_fn_group (fn_groups, name);
  fn_field fnf;
  pdb_fill_fn_field (pdb, &fnf, attr, type_ti);
  grp->methods.push_back (fnf);
}

/* LF_METHOD — an overloaded member function (multiple overloads).
   Reads the referenced LF_METHODLIST (layout: LF_MLIST_* offsets) and
   adds each overload to the same fn_group.  */
static void
pdb_parse_lf_method (pdb_per_objfile *pdb, const gdb_byte *p,
		     const gdb_byte *end, std::vector<pdb_fn_group> &fn_groups)
{
  uint16_t mcount = read_u16 (p + LF_METHOD_COUNT_OFFS);
  uint32_t mlist_ti = read_u32 (p + LF_METHOD_MLIST_OFFS);
  auto name = CSTR (p + LF_METHOD_NAME_OFFS);

  const pdb_tpi_type *ml = pdb_tpi_get_type (&pdb->tpi, mlist_ti);
  if (ml == nullptr || ml->leaf != LF_METHODLIST)
    return;

  pdb_fn_group *grp = pdb_find_or_add_fn_group (fn_groups, name);

  /* Walk the LF_METHODLIST entries.  Each is 8 bytes (attr + pad + type_ti),
     optionally followed by 4 bytes (vbaseoff) for introducing virtuals.  */
  const gdb_byte *mp = ml->data;
  const gdb_byte *mend = ml->data + ml->data_len;
  for (uint16_t i = 0; i < mcount && mp + LF_MLIST_ENTRY_SIZE <= mend; i++)
    {
      uint16_t mattr = read_u16 (mp + LF_MLIST_ATTR_OFFS);
      uint32_t mtype_ti = read_u32 (mp + LF_MLIST_TYPE_OFFS);
      mp += LF_MLIST_ENTRY_SIZE;

      if (CV_MPROP_HAS_VBASEOFF (CV_MPROP (mattr)))
	mp += 4;

      fn_field fnf;
      pdb_fill_fn_field (pdb, &fnf, mattr, mtype_ti);
      grp->methods.push_back (fnf);
    }
}

/* LF_VBCLASS / LF_IVBCLASS — virtual or indirect-virtual base class.
   Field layout defined by LF_VBCLASS_* offsets in pdb-internal.h.

   The record carries two trailing numeric leaves we currently discard:
     vbpoff — byte offset of the virtual base pointer (vbptr) inside the
	      derived object.
     vbte   — byte offset into the vbtable pointed to by the vbptr; the
	      slot at that offset holds the displacement from the vbptr
	      to the virtual base sub-object.

   Runtime address of the virtual base sub-object is then:
     vbase_addr = derived_addr
		  + vbpoff
		  + *(ptrdiff_t *)(*(void **)(derived_addr + vbpoff)
				   + vbte)

   GDB cannot evaluate this today: the arithmetic lives in a per-ABI
   `virtual_base_offset' hook (see gnu-v3-abi.c for the Itanium/g++ case),
   and no MS C++ ABI implementation exists in GDB.  Until one is added
   (and vbpoff/vbte are stashed on the field), `f.set_virtual ()' below
   only flags the base for `ptype' display; printing/dereferencing the
   virtual base on an MSVC binary will not work.  */
static void
pdb_parse_lf_vbclass (pdb_per_objfile *pdb, const gdb_byte *p,
		      const gdb_byte *end [[maybe_unused]],
		      std::vector<field> &baseclasses)
{
  uint16_t attr = read_u16 (p + LF_VBCLASS_ATTR_OFFS);
  uint32_t base_ti = read_u32 (p + LF_VBCLASS_TYPE_OFFS);

  type *base_type = pdb_tpi_resolve_type_internal (pdb, base_ti);
  field f;
  memset (&f, 0, sizeof (f));
  f.set_type (base_type);
  f.set_name (base_type->name () ? base_type->name () : "");
  f.set_loc_bitpos (0);
  f.set_virtual ();
  pdb_set_field_accessibility (&f, attr);
  baseclasses.push_back (f);
}

/* ---------------------------------------------------------------
   LF_FIELDLIST top-level parser.

   Walks all sub-records in the fieldlist, dispatching to the helpers
   above.  Collects results into separate vectors, then attaches
   everything to the GDB type.

   Mirrors dwarf2_add_field / dwarf2_attach_fields_to_type /
   dwarf2_attach_fn_fields_to_type from dwarf2/read.c.
   --------------------------------------------------------------- */
static void
pdb_tpi_parse_fieldlist (pdb_per_objfile *pdb, type *type,
			 uint32_t fieldlist_ti,
			 uint16_t expected_count [[maybe_unused]])
{
  const pdb_tpi_type *fl = pdb_tpi_get_type (&pdb->tpi, fieldlist_ti);
  if (fl == nullptr || fl->leaf != LF_FIELDLIST)
    {
      pdb_warning ("LF_FIELDLIST: type index 0x%x invalid or wrong leaf",
		   fieldlist_ti);
      return;
    }

  const gdb_byte *p = fl->data;
  const gdb_byte *end = fl->data + fl->data_len;
  bool is_enum = (type->code () == TYPE_CODE_ENUM);

  /* Four output buckets (attached to the type at the end).  */
  std::vector<field> baseclasses;
  std::vector<field> fields;
  std::vector<decl_field> nested_types;
  std::vector<pdb_fn_group> fn_groups;

  while (p < end)
    {
      /* Skip 1-3 alignment padding bytes between sub-records.  */
      p = pdb_fieldlist_skip_padding (p, end);
      /* Need at least 2 bytes for the sub-record leaf type.  */
      if (p + 2 > end)
	break;

      uint16_t leaf = read_u16 (p);

      /* First compute the pointer to the next sub-record. If there is a problem
	 we just bail out. This makes it easier to check for the correctness of
	 the record before parsing it */
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
	  /* Fieldlist continuation: when a fieldlist exceeds certain size (?),
	     MSVC splits it into multiple LF_FIELDLIST records chained by
	     LF_INDEX sub-records.  Switch to the continuation record.  */
	  uint32_t cont_ti = read_u32 (p + LF_INDEX_TYPE_OFFS);
	  const pdb_tpi_type *cont = pdb_tpi_get_type (&pdb->tpi, cont_ti);
	  if (cont == nullptr || cont->leaf != LF_FIELDLIST)
	    {
	      pdb_warning ("LF_INDEX: continuation 0x%x invalid", cont_ti);
	      break;
	    }

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
	    case LF_IVBCLASS:
	      pdb_parse_lf_vbclass (pdb, p, end, baseclasses);
	      break;
	    case LF_VFUNCTAB:
	      /* Virtual function table pointer — nothing added to GDB.  */
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

  /* Move collected vectors into the GDB type.  */

  uint32_t nbaseclasses = baseclasses.size ();
  uint32_t nfields = nbaseclasses + fields.size ();

  type->alloc_fields (nfields);

  /* Base classes come first.  */
  uint32_t i = 0;
  for (const auto &f : baseclasses)
    type->field (i++) = f;
  for (const auto &f : fields)
    type->field (i++) = f;

  /* Allocate cplus_struct_type if we have any C++ metadata.  */
  if (nbaseclasses > 0 || !nested_types.empty () || !fn_groups.empty ())
    {
      ALLOCATE_CPLUS_STRUCT_TYPE (type);
    }

  if (nbaseclasses > 0)
    TYPE_N_BASECLASSES (type) = nbaseclasses;

  /* Attach nested types to cplus_struct_type.  */
  if (!nested_types.empty ())
    {
      uint32_t count = nested_types.size ();
      TYPE_NESTED_TYPES_ARRAY (type)
	= (decl_field *) TYPE_ALLOC (type, count * sizeof (decl_field));
      TYPE_NESTED_TYPES_COUNT (type) = count;
      for (uint32_t j = 0; j < count; j++)
	TYPE_NESTED_TYPES_FIELD (type, j) = nested_types[j];
    }

  /* Attach member function groups to cplus_struct_type.  */
  if (!fn_groups.empty ())
    {
      auto ngroups = fn_groups.size ();
      TYPE_FN_FIELDLISTS (type)
	= (fn_fieldlist *) TYPE_ALLOC (type, ngroups * sizeof (fn_fieldlist));
      TYPE_NFN_FIELDS (type) = static_cast<short> (ngroups);

      for (uint32_t j = 0; j < ngroups; j++)
	{
	  auto &grp = fn_groups[j];
	  TYPE_FN_FIELDLIST_NAME (type, j) = grp.name;
	  TYPE_FN_FIELDLIST_LENGTH (type, j) = grp.methods.size ();
	  TYPE_FN_FIELDLIST1 (type, j) = (fn_field *)
	    TYPE_ALLOC (type, grp.methods.size () * sizeof (fn_field));
	  for (uint32_t k = 0; k < grp.methods.size (); k++)
	    TYPE_FN_FIELDLIST1 (type, j)[k] = grp.methods[k];
	}
    }
}

/* Parsed fields common to all compound type records.  */
struct pdb_compound_fields
{
  uint16_t count;        /* Number of members.  */
  uint16_t property;     /* CV_prop_t flags.  */
  uint32_t fieldlist_ti; /* Type index of LF_FIELDLIST.  */
  uint64_t byte_size;    /* Struct/union size, or underlying type length.  */
  const char *name;
};

/* Parse common fields from LF_STRUCTURE/CLASS/UNION records. Field layout is
   defined by LF_STRUCT_* / LF_UNION_* offsets in pdb-internal.h.  */
static bool
pdb_parse_tagged_record (const pdb_tpi_type *rec, uint32_t min_size,
			 uint32_t fieldlist_offs, uint32_t data_offs,
			 const char *name, pdb_compound_fields *out)
{
  if (!pdb_check_record_size (rec, min_size, name))
    return false;

  out->count = read_u16 (rec->data + LF_STRUCT_COUNT_OFFS);
  out->property = read_u16 (rec->data + LF_STRUCT_PROPERTY_OFFS);
  out->fieldlist_ti = read_u32 (rec->data + fieldlist_offs);

  /* The struct/union byte size is variable-length encoded (numeric leaf)  */
  const gdb_byte *data_ptr = rec->data + data_offs;
  uint32_t remaining = rec->data_len - data_offs;

  uint32_t nr = pdb_cv_read_numeric (data_ptr, remaining, &out->byte_size);
  if (nr == 0)
    return false;

  out->name = CSTR (data_ptr + nr);
  return true;
}

/* GDB type assembly for compound types (struct/union/enum).
   Allocates a new type, sets code/name/length and parses the field list.
   For forward references marks the type as a stub and skips field list parsing.
   Mirrors read_enumeration_type, read_structure_type ...  in dwarf2/read.c'  */
static type *
pdb_tpi_init_compound (pdb_per_objfile *pdb, enum type_code code,
		       const pdb_compound_fields *f,
		       uint32_t type_idx [[maybe_unused]])
{
  type_allocator alloc (pdb->objfile, language_c);
  type *type = alloc.new_type ();

  type->set_code (code);

  if (f->name != nullptr && f->name[0] != '\0')
    type->set_name (f->name);

  type->set_length (f->byte_size);

  /* Early caching seems unnecessary: the TPI stream is a topologically sorted
     DAG, so any type that could create a circular reference has a forward-ref
     stub emitted at a lower index.  */
  //if (pdb->tpi.type_cache != nullptr)
  //  pdb->tpi.type_cache[type_idx] = type;

  /* Forward reference — mark as stub.
     Mirrors DW_AT_declaration handling in dwarf2/read.c.  */
  if (f->property & CV_PROP_FWDREF)
    {
      type->set_is_stub (true);
      return type;
    }

  /* Finally parse all the fields  */
  if (f->fieldlist_ti != 0)
    pdb_tpi_parse_fieldlist (pdb, type, f->fieldlist_ti, f->count);

  return type;
}

/* LF_STRUCTURE / LF_CLASS  */
static type *
pdb_tpi_make_struct (pdb_per_objfile *pdb, const pdb_tpi_type *rec,
		     uint32_t type_idx)
{
  pdb_compound_fields f;
  if (!pdb_parse_tagged_record (rec, LF_STRUCT_MIN_SIZE,
				LF_STRUCT_FIELDLIST_OFFS, LF_STRUCT_DATA_OFFS,
				"LF_STRUCTURE/CLASS", &f))
    return pdb_tpi_get_unsupported_type (pdb);

  type *type = pdb_tpi_init_compound (pdb, TYPE_CODE_STRUCT, &f, type_idx);

  if (rec->leaf == LF_CLASS)
    type->set_is_declared_class (true);

  return type;
}

/* LF_UNION.  */
static type *
pdb_tpi_make_union (pdb_per_objfile *pdb, const pdb_tpi_type *rec,
		    uint32_t type_idx)
{
  pdb_compound_fields f;
  if (!pdb_parse_tagged_record (rec, LF_UNION_MIN_SIZE,
				LF_UNION_FIELDLIST_OFFS, LF_UNION_DATA_OFFS,
				"LF_UNION", &f))
    return pdb_tpi_get_unsupported_type (pdb);

  return pdb_tpi_init_compound (pdb, TYPE_CODE_UNION, &f, type_idx);
}

/* LF_ENUM.  */
static type *
pdb_tpi_make_enum (pdb_per_objfile *pdb, const pdb_tpi_type *rec,
		   uint32_t type_idx)
{
  if (!pdb_check_record_size (rec, LF_ENUM_MIN_SIZE, "LF_ENUM"))
    return pdb_tpi_get_unsupported_type (pdb);

  /* Get underlaying type  */
  uint32_t utype_ti = read_u32 (rec->data + LF_ENUM_UTYPE_OFFS);
  type *enum_type = pdb_tpi_resolve_type_internal (pdb, utype_ti);

  auto name = CSTR (rec->data + LF_ENUM_NAME_OFFS);
  const auto nul = (const gdb_byte *) memchr (rec->data + LF_ENUM_NAME_OFFS, 0,
					      rec->data_len
						- LF_ENUM_NAME_OFFS);
  if (nul == nullptr)
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

  return type;
}

/* Recursive type resolver with caching.
   Compound types dispatch to pdb_tpi_make_* helpers.
   Unsupported leaf types return TYPE_CODE_ERROR.  */
static type *
pdb_tpi_resolve_type_internal (pdb_per_objfile *pdb, uint32_t type_idx)
{
  /* Simple / built-in types.  */
  if (CV_TI_IS_SIMPLE (type_idx))
    return pdb_tpi_resolve_simple_type (pdb, type_idx);

  /* Check cache  */
  if (pdb->tpi.type_cache[type_idx] != nullptr)
    return pdb->tpi.type_cache[type_idx];

  /* Look up compound type record from TPI stream.  */
  const pdb_tpi_type *rec = pdb_tpi_get_type (&pdb->tpi, type_idx);
  if (rec == nullptr)
    return pdb_tpi_get_unsupported_type (pdb);

  type *result = nullptr;

  switch (rec->leaf)
    {
    case LF_MODIFIER:
      result = pdb_tpi_make_modifier (pdb, rec);
      break;
    case LF_PROCEDURE:
      result = pdb_tpi_make_procedure (pdb, rec);
      break;
    case LF_MFUNCTION:
      result = pdb_tpi_make_mfunction (pdb, rec);
      break;
    case LF_POINTER:
      result = pdb_tpi_make_pointer (pdb, rec);
      break;
    case LF_ARRAY:
      result = pdb_tpi_make_array (pdb, rec);
      break;
    case LF_BITFIELD:
      result = pdb_tpi_make_bitfield (pdb, rec);
      break;
    case LF_STRUCTURE:
    case LF_CLASS:
      result = pdb_tpi_make_struct (pdb, rec, type_idx);
      break;
    case LF_UNION:
      result = pdb_tpi_make_union (pdb, rec, type_idx);
      break;
    case LF_ENUM:
      result = pdb_tpi_make_enum (pdb, rec, type_idx);
      break;
    case LF_ARGLIST:
    case LF_FIELDLIST:
    case LF_VTSHAPE:
    case LF_LABEL:
    case LF_METHODLIST:
      result = builtin_type (pdb->objfile->arch ())->builtin_void;
      break;
    default:
      /* Unsupported leaf type  */
      break;
    }

  if (result == nullptr)
    result = pdb_tpi_get_unsupported_type (pdb);

  /* Cache the result.  */
  if (pdb->tpi.type_cache != nullptr)
    pdb->tpi.type_cache[type_idx] = result;

  return result;
}

/* See pdb-internal.h.  */
type *
pdb_tpi_resolve_type (pdb_per_objfile *pdb, uint32_t type_idx)
{
  return pdb_tpi_resolve_type_internal (pdb, type_idx);
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
      /* Add 1 for implicit 'this' parameter.  */
      return read_u16 (rec->data + LF_MFUNC_PARMCOUNT_OFFS) + 1;
    }

  else if (rec->leaf == LF_PROCEDURE && rec->data_len >= LF_PROC_SIZE)
    {
      return read_u16 (rec->data + LF_PROC_PARMCOUNT_OFFS);
    }

  return 0;
}

/* See pdb-internal.h.  */

bool
pdb_tpi_type_is_fwdref (pdb_tpi_context const *tpi, uint32_t type_idx)
{
  const pdb_tpi_type *rec = pdb_tpi_get_type (tpi, type_idx);
  if (rec == nullptr)
    return false;

  uint16_t prop;

  switch (rec->leaf)
    {
    case LF_CLASS:
    case LF_STRUCTURE:
      if (rec->data_len < LF_STRUCT_PROPERTY_OFFS + 2)
	return false;
      prop = read_u16 (rec->data + LF_STRUCT_PROPERTY_OFFS);
      break;

    case LF_UNION:
      if (rec->data_len < LF_UNION_PROPERTY_OFFS + 2)
	return false;
      prop = read_u16 (rec->data + LF_UNION_PROPERTY_OFFS);
      break;

    case LF_ENUM:
      if (rec->data_len < LF_ENUM_PROPERTY_OFFS + 2)
	return false;
      prop = read_u16 (rec->data + LF_ENUM_PROPERTY_OFFS);
      break;

    default:
      return false;
    }

  return (prop & CV_PROP_FWDREF) != 0;
}

/* See pdb-internal.h.  */
void
pdb_register_tpi_typedefs (pdb_per_objfile *pdb)
{
  const pdb_tpi_context *tpi = &pdb->tpi;
  if (tpi->types == nullptr || tpi->type_idx_end <= tpi->type_idx_begin)
    return;

  auto *cu = new buildsym_compunit (pdb->objfile, "<pdb-types>", "",
				    language_cplus, 0);
  uint32_t count = tpi->type_idx_end - tpi->type_idx_begin;
  for (uint32_t i = 0; i < count; i++)
    {
      const pdb_tpi_type *rec = &tpi->types[i];
      uint32_t ti = tpi->type_idx_begin + i;
      const char *name = nullptr;

      /* Skip forward references — we only want complete definitions.  */
      if (pdb_tpi_type_is_fwdref (&pdb->tpi, ti))
	continue;

      switch (rec->leaf)
	{
	case LF_CLASS:
	case LF_STRUCTURE:
	  if (rec->data_len > LF_STRUCT_MIN_SIZE)
	    {
	      uint64_t size_val;
	      uint32_t nr
		= pdb_cv_read_numeric (rec->data + LF_STRUCT_DATA_OFFS,
				       rec->data_len - LF_STRUCT_DATA_OFFS,
				       &size_val);
	      if (nr > 0 && LF_STRUCT_DATA_OFFS + nr < rec->data_len)
		name = CSTR (rec->data + LF_STRUCT_DATA_OFFS + nr);
	    }

	  break;

	case LF_UNION:
	  if (rec->data_len > LF_UNION_MIN_SIZE)
	    {
	      uint64_t size_val;
	      uint32_t nr
		= pdb_cv_read_numeric (rec->data + LF_UNION_DATA_OFFS,
				       rec->data_len - LF_UNION_DATA_OFFS,
				       &size_val);
	      if (nr > 0 && LF_UNION_DATA_OFFS + nr < rec->data_len)
		name = CSTR (rec->data + LF_UNION_DATA_OFFS + nr);
	    }

	  break;

	case LF_ENUM:
	  if (rec->data_len > LF_ENUM_MIN_SIZE)
	    name = CSTR (rec->data + LF_ENUM_NAME_OFFS);

	  break;

	default:
	  break;
	}

      if (name != nullptr && name[0] != '\0')
	{
	  type *gdb_type = pdb_tpi_resolve_type (pdb, ti);
	  if (gdb_type != nullptr)
	    {
	      auto *sym = new (&pdb->objfile->objfile_obstack) symbol;
	      sym->set_language (language_cplus,
				 &pdb->objfile->objfile_obstack);
	      sym->compute_and_set_names (name, true, pdb->objfile->per_bfd);
	      sym->set_domain (STRUCT_DOMAIN);
	      sym->set_loc_class_index (LOC_TYPEDEF);
	      sym->set_type (gdb_type);
	      add_symbol_to_list (sym, cu->get_global_symbols ());
	    }
	}
    }

  cu->end_compunit_symtab (0);
  delete cu;
}

} // namespace pdb
