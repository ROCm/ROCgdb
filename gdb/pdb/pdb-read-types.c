/* PDB type reader.

   Copyright (C) 2026 Free Software Foundation, Inc.
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
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

/* This file implements TPI (Type Program Information) stream parsing
   and type resolution for use with GDB symbols.

   References:
     - microsoft-pdb: https://github.com/microsoft/microsoft-pdb
     - LLVM docs: https://llvm.org/docs/PDB/TpiStream.html

   CURRENT LIMITATIONS:
   Not all types are supported (classes, templates...).
   Unsupported types are mapped to TYPE_CODE_ERROR and display as
   "<unsupported PDB type>".
*/

#include "symtab.h"
#include "gdbtypes.h"
#include "objfiles.h"
#include "complaints.h"
#include "pdb/pdb.h"
#include <string.h>

/* ---------------------------------------------------------------
   TPI / IPI Stream (Type Program Information)
   https://llvm.org/docs/PDB/TpiStream.html

   The TPI stream (stream 2) and IPI stream (stream 4) contain type records that
   describe all types used by the program.  Symbols reference types through a
   32-bit Type Index (TI).  Type indices < 0x1000 are built-in types whose
   meaning is encoded directly in the index value:
     bits  0-7  : SimpleTypeKind (void, int, float, …)
     bits  8-11 : SimpleTypeMode (direct, near ptr, far ptr, …)
   Any index >= TypeIndexBegin (always 0x1000 in practice) refers to a record in
   the TPI (or IPI) type record array.  Records form a topologically sorted DAG:
   record B may only reference record A if A's type index < B's type index.
   TPI/IPI stream layout:
     TpiStreamHeader  (56 bytes)
     Array of Type Records:
       Type Record Layout:
	  RecordLen (2 bytes)  — length of RecordKind + variable data
	  RecordKind (2 bytes) — Leaf type (LF_*)
	  RecordData (RecordLen - 2 bytes) — fields depend on RecordKind.
   ---------------------------------------------------------------.  */

/* TPI Stream Header defines used during parsing.  */
#define TPI_HDR_VERSION_OFFS             0   /* Version (4-byte).  */
#define TPI_HDR_HEADER_SIZE_OFFS         4   /* HeaderSize (4-byte).  */
#define TPI_HDR_TYPE_INDEX_BEGIN_OFFS    8   /* TypeIndexBegin (4-byte).  */
#define TPI_HDR_TYPE_INDEX_END_OFFS     12   /* TypeIndexEnd (4-byte).  */
#define TPI_HDR_TYPE_REC_BYTES_OFFS     16   /* TypeRecordBytes (4-byte).  */
#define TPI_HDR_SIZE                    56   /* Total header size.  */

#define TPI_VERSION_V80  20040203            /* Only observed version.  */

/* Types (so called leaf types) from microsoft-pdb/cvinfo.h.  */
#define LF_NUMERIC       0x8000
#define LF_CHAR          0x8000
#define LF_SHORT         0x8001
#define LF_USHORT        0x8002
#define LF_LONG          0x8003
#define LF_ULONG         0x8004
#define LF_QUADWORD      0x8009
#define LF_UQUADWORD     0x800a
#define LF_MODIFIER      0x1001
#define LF_POINTER       0x1002
#define LF_ARRAY         0x1503
#define LF_ARGLIST       0x1201
#define LF_FIELDLIST     0x1203
#define LF_BITFIELD      0x1205
#define LF_PROCEDURE     0x1008
#define LF_MFUNCTION     0x1009
#define LF_VTSHAPE       0x0a
#define LF_LABEL         0x0e
#define LF_METHODLIST    0x1206

/* LF_PROCEDURE (0x1008) field offsets within record data.
       uint32_t rvtype      (0)  return type index
       uint8_t  calltype    (4)  calling convention
       uint8_t  funcattr    (5)  attributes
       uint16_t parmcount   (6)  number of parameters
       uint32_t arglist     (8)  type index of argument list  */
#define LF_PROC_RVTYPE_OFFS     0
#define LF_PROC_PARMCOUNT_OFFS  6
#define LF_PROC_ARGLIST_OFFS    8
#define LF_PROC_SIZE           12   /* Minimum record data size.  */

/* LF_MFUNCTION (0x1009) field offsets within record data.
       uint32_t rvtype      (0)  return type index
       uint32_t classtype   (4)  containing class type index
       uint32_t thistype    (8)  this pointer type index
       uint8_t  calltype   (12)  calling convention
       uint8_t  funcattr   (13)  attributes
       uint16_t parmcount  (14)  number of parameters
       uint32_t arglist    (16)  type index of argument list
       int32_t  thisadjust (20)  this adjuster  */
#define LF_MFUNC_RVTYPE_OFFS     0
#define LF_MFUNC_CLASSTYPE_OFFS  4
#define LF_MFUNC_THISTYPE_OFFS   8
#define LF_MFUNC_PARMCOUNT_OFFS 14
#define LF_MFUNC_ARGLIST_OFFS   16
#define LF_MFUNC_SIZE           24   /* Minimum record data size.  */

/* LF_MODIFIER (0x1001).  */

/* CV_modifier_e: Modifier attribute bits in LF_MODIFIER attr.  */
#define CV_MODIFIER_CONST      0x01  /* const modifier.  */
#define CV_MODIFIER_VOLATILE   0x02  /* volatile modifier.  */

/* LF_POINTER (0x1002).  */

/* Pointer/reference mode encoding in LF_POINTER attr.  */
#define CV_PTRMODE_LVALUE_REF  1  /* Lvalue reference (&).  */
#define CV_PTRMODE_POINTER     2  /* Pointer (*).  */
#define CV_PTRMODE_RVALUE_REF  3  /* Rvalue reference (&&).  */

typedef struct
{
  unsigned int ptrtype     : 5;  /* Ordinal specifying pointer type (CV_ptrtype_e).  */
  unsigned int ptrmode     : 3;  /* Ordinal specifying pointer mode (CV_ptrmode_e).  */
  unsigned int isflat32    : 1;  /* True if 0:32 pointer.  */
  unsigned int isvolatile  : 1;  /* TRUE if volatile pointer.  */
  unsigned int isconst     : 1;  /* TRUE if const pointer.  */
  unsigned int isunaligned : 1;  /* TRUE if unaligned pointer.  */
  unsigned int isrestrict  : 1;  /* TRUE if restricted pointer.  */
  unsigned int size        : 6;  /* Size of pointer (in bytes).  */
  unsigned int ismocom     : 1;  /* TRUE if MoCOM pointer (^ or %).  */
  unsigned int islref      : 1;  /* TRUE if & ref-qualifier.  */
  unsigned int isrref      : 1;  /* TRUE if && ref-qualifier.  */
  unsigned int unused      : 10; /* Pad out to 32-bits.  */
} pdb_lf_pointer_attr;

typedef struct
{
  uint32_t utype;            /* Type index of underlying type.  */
  pdb_lf_pointer_attr attr;  /* Attributes bitfield.  */
} pdb_lf_pointer;


/* LF_ARRAY (0x1503).  */

typedef struct
{
  uint32_t elemtype;  /* Type index of element type.  */
  uint32_t idxtype;   /* Type index of indexing type.  */
  unsigned char data[];  /* Variable-length: numeric leaf for size, then name.  */
} pdb_lf_array;

/* Forward declarations.  */
static struct type *pdb_tpi_resolve_type_internal (struct pdb_per_objfile *pdb,
						   uint32_t type_idx);
static const struct pdb_tpi_type *pdb_tpi_get_type (struct pdb_tpi_context *tpi,
						    uint32_t type_idx);
uint32_t pdb_cv_read_numeric (const gdb_byte *data, uint32_t max_len,
			     uint64_t *value);
static struct type *pdb_tpi_get_unsupported_type (struct pdb_per_objfile *pdb);

/* Get a type record from the TPI context by type index.
   Returns nullptr if the index is out of range.  */
static const struct pdb_tpi_type *
pdb_tpi_get_type (struct pdb_tpi_context *tpi, uint32_t type_idx)
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
pdb_parse_tpi_records (struct pdb_per_objfile *pdb,
		       const gdb_byte *records, uint32_t rec_bytes,
		       const char *stream_name,
		       struct pdb_tpi_context &tpi)
{
  uint32_t num_types = tpi.type_idx_end - tpi.type_idx_begin;

  /* Allocate type record array on obstack - will be accessed
     during any module's expansion, type by type...  */
  struct pdb_tpi_type *types = nullptr;
  if (num_types > 0)
    types = OBSTACK_CALLOC (&pdb->objfile->objfile_obstack, num_types,
			    struct pdb_tpi_type);

  /* Walk the records and parse into tpi->types. */
  const gdb_byte *p = records;
  const gdb_byte *rec_end = p + rec_bytes;
  uint32_t idx = 0;

  while (idx < num_types)
    {
      if (p + CV_REC_HDR_SIZE > rec_end)
	{
	  pdb_error ("%s: unexpected end of records at type 0x%x",
		     stream_name, tpi.type_idx_begin + idx);
	  break;
	}
      auto rec_len = UINT16_CAST (p);
      auto rec_kind = UINT16_CAST (p + 2);

      if (rec_len < 2)
	{
	  pdb_error ("%s: type 0x%x has invalid RecordLen %u",
		     stream_name, tpi.type_idx_begin + idx, rec_len);
	  break;
	}

      /* rec_len includes the 2-byte RecordKind but not the 2-byte RecordLen
	 field itself.  Total on-disk size = 2 (RecordLen) + rec_len.  */
      auto rec_size = rec_len + 2;

      if (p + rec_size > rec_end)
	{
	  pdb_error ("%s: type 0x%x overflows record area",
		     stream_name, tpi.type_idx_begin + idx);
	  break;
	}

      types[idx].leaf = rec_kind;
      types[idx].length = rec_len;
      types[idx].data = p + CV_REC_HDR_SIZE;
      types[idx].data_len = rec_len - 2;

      idx++;

      /* Advance to next record (4-byte aligned). */
      p += rec_size;
      p = (const gdb_byte *) align_up((uintptr_t)p, 4);
    }

  tpi.types = types;
  return true;
}

/* Read and parse TPI (stream 2) if present.
   Populates pdb->tpi->types[] with decoded type records.  */
bool
pdb_read_tpi_stream (struct pdb_per_objfile *pdb)
{
  auto &tpi = pdb->tpi;

  if ( pdb->num_streams <= PDB_STREAM_TPI
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
      return false;
    }

  gdb_byte *tpi_data
    = pdb->stream_data[PDB_STREAM_TPI] = tpi_buf.release ();

  pdb->tpi_size = pdb->stream_sizes[PDB_STREAM_TPI];

  if (pdb->tpi_size < TPI_HDR_SIZE)
    {
      pdb_error ("TPI stream too small (%u bytes)", pdb->tpi_size);
      return false;
    }

  uint32_t version  = UINT32_CAST (tpi_data + TPI_HDR_VERSION_OFFS);
  uint32_t hdr_size = UINT32_CAST (tpi_data + TPI_HDR_HEADER_SIZE_OFFS);
  uint32_t rec_bytes = UINT32_CAST (tpi_data + TPI_HDR_TYPE_REC_BYTES_OFFS);
  tpi.type_idx_begin = UINT32_CAST (tpi_data + TPI_HDR_TYPE_INDEX_BEGIN_OFFS);
  tpi.type_idx_end = UINT32_CAST (tpi_data + TPI_HDR_TYPE_INDEX_END_OFFS);

  if (version != TPI_VERSION_V80)
    {
      pdb_error ("TPI unknown version 0x%08x (expected 0x%08x)",
		 version, TPI_VERSION_V80);
      return false;
    }

  if (hdr_size < TPI_HDR_SIZE)
    {
      pdb_error ("TPI header size %u too small", hdr_size);
      return false;
    }

  if (hdr_size + rec_bytes > pdb->tpi_size)
    {
      pdb_error ("TPI records overflow stream (%u + %u > %u)",
		 hdr_size, rec_bytes, pdb->tpi_size);
      return false;
    }

  /* Log TPI stream info before parsing.  */
  uint32_t num_types = tpi.type_idx_end - tpi.type_idx_begin;
  pdb_dbg_printf ("TPI: TI=[0x%x..0x%x)  num_types=%u  rec_bytes=%u",
		  tpi.type_idx_begin, tpi.type_idx_end,
		  num_types, rec_bytes);

  if (!pdb_parse_tpi_records (pdb, tpi_data + hdr_size, rec_bytes, "TPI", tpi))
    {
      return false;
    }

  /* Allocate unified type cache covering simple types (0x0000-0x0FFF)
     and all compound types up to type_idx_end.  */
  uint32_t cache_size = tpi.type_idx_end > 0x1000 ? tpi.type_idx_end : 0x1000;
  pdb->tpi.type_cache = OBSTACK_CALLOC (&pdb->objfile->objfile_obstack,
					 cache_size, struct type *);

  return true;
}

/* Resolve a simple (built-in) type index to a GDB type.
 Simple type indices are < 0x1000 and encode:
     bits 0-7  : SimpleTypeKind
     bits 8-11 : SimpleTypeMode (direct, near pointer, etc.).
   Results are cached in pdb_tpi_context to avoid recreating custom-sized pointers.  */
static struct type *
pdb_tpi_resolve_simple_type (struct pdb_per_objfile *pdb, uint32_t type_idx)
{
  /* Check cache first.  */
  if (pdb->tpi.type_cache != nullptr)
    {
      struct type *cached = pdb->tpi.type_cache[type_idx];
      if (cached != nullptr)
	return cached;
    }

  struct gdbarch *garch = pdb->objfile->arch ();
  const struct builtin_type *bt = builtin_type (garch);

  uint32_t kind = CV_SIMPLE_KIND (type_idx);
  uint32_t mode = CV_SIMPLE_MODE (type_idx);

  /* First resolve the base type from the Kind field.  */
  struct type *base = nullptr;

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

    case CV_INT64:
      base = bt->builtin_int64;
      break;

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
      //pdb_warning ("Unknown simple type 0x%02x in TI 0x%04x", kind, type_idx);
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
  struct type *ptr_type;

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

/* Create a custom UNDEFINED struct type for unsupported types.
   Cached per-objfile in pdb_tpi_context::undefined_type.  */
static struct type *
pdb_tpi_get_unsupported_type (struct pdb_per_objfile *pdb)
{
  if (pdb->tpi.undefined_type != nullptr)
    return pdb->tpi.undefined_type;

  /* Create an error type for unsupported/unresolved PDB types.
     TYPE_CODE_ERROR tells GDB's value-printing infrastructure to print
     "<error type>" instead of trying to interpret the value as a struct,
     which would crash.  */
  type_allocator alloc (pdb->objfile->arch ());
  struct type *undef = alloc.new_type (TYPE_CODE_ERROR, 8, nullptr);
  undef->set_name ("<unsupported PDB type>");

  pdb->tpi.undefined_type = undef;
  return undef;
}

/* Read a CodeView numeric leaf — a variable-length encoded integer used
   for array sizes, dimensions, and other numeric fields in type records.

   Returns: Number of bytes consumed (0 if error or invalid).
   Output:  *value = the decoded numeric value.

   Caller must use the return value to advance the data pointer to skip
   past this leaf if parsing sequential fields.  */
uint32_t
pdb_cv_read_numeric (const gdb_byte *data, uint32_t max_len, uint64_t *value)
{
  if (max_len < 2)
    return 0;

  auto leaf = UINT16_CAST (data);

  /* Simple leaf value - encoded directly in the leaf (2 bytes) */
  if (leaf < LF_NUMERIC)
    {
      *value = leaf;
      return 2;
    }

  switch (leaf)
    {
    case LF_CHAR:
      if (max_len < 3) return 0;
      *value = INT8_CAST (data + 2);
      return 3;
    case LF_SHORT:
      if (max_len < 4) return 0;
      *value = INT16_CAST (data + 2);
      return 4;
    case LF_USHORT:
      if (max_len < 4) return 0;
      *value = UINT16_CAST (data + 2);
      return 4;
    case LF_LONG:
      if (max_len < 6) return 0;
      *value = INT32_CAST (data + 2);
      return 6;
    case LF_ULONG:
      if (max_len < 6) return 0;
      *value = UINT32_CAST (data + 2);
      return 6;
    case LF_QUADWORD:
      if (max_len < 10) return 0;
      *value = UINT64_CAST (data + 2);
      return 10;
    case LF_UQUADWORD:
      if (max_len < 10) return 0;
      *value = UINT64_CAST (data + 2);
      return 10;
    default:
      pdb_dbg_printf ("invalid/truncated numeric leaf 0x%04x", leaf);
      return 0;
    }
}

/* LF_MODIFIER  */
static struct type *
pdb_tpi_make_modifier (struct pdb_per_objfile *pdb,
		       const struct pdb_tpi_type *rec)
{
  if (rec->data_len < 6)
    {
      pdb_warning ("LF_MODIFIER record truncated (got %u)", rec->data_len);
      return pdb_tpi_get_unsupported_type (pdb);
    }

  auto type_idx = UINT32_CAST (rec->data);
  auto attr = UINT16_CAST (rec->data + 4);

  auto *base = pdb_tpi_resolve_type_internal (pdb, type_idx);

  bool is_const = (attr & CV_MODIFIER_CONST) != 0;
  bool is_volatile = (attr & CV_MODIFIER_VOLATILE) != 0;

  if (is_const || is_volatile)
    base = make_cv_type (is_const, is_volatile, base);

  return base;
}

/* LF_POINTER  */
static struct type *
pdb_tpi_make_pointer (struct pdb_per_objfile *pdb,
		      const struct pdb_tpi_type *rec)
{
  if (rec->data_len < sizeof (pdb_lf_pointer))
    {
      pdb_warning ("LF_POINTER record truncated (expected >= %zu, got %u)",
		   sizeof (pdb_lf_pointer), rec->data_len);
      return pdb_tpi_get_unsupported_type (pdb);
    }

  const pdb_lf_pointer *ptr = (const pdb_lf_pointer *) rec->data;

  auto *pointee = pdb_tpi_resolve_type_internal (pdb, ptr->utype);

  struct type *result;
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
static struct type *
pdb_tpi_make_array (struct pdb_per_objfile *pdb,
		    const struct pdb_tpi_type *rec)
{
  if (rec->data_len < sizeof (pdb_lf_array))
    {
      pdb_warning ("LF_ARRAY record truncated (expected >= %zu, got %u)",
		   sizeof (pdb_lf_array), rec->data_len);
      return pdb_tpi_get_unsupported_type (pdb);
    }

  const pdb_lf_array *array = (const pdb_lf_array *) rec->data;

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
  auto *range_type = create_static_range_type (alloc, idx_type,
						 0, num_elements > 0
						 ? num_elements - 1 : 0);
  auto *array_type = create_array_type (alloc, elem_type, range_type);
  array_type->set_length (total_size);
  return array_type;
}

/* LF_BITFIELD (0x1205).  */
static struct type *
pdb_tpi_make_bitfield (struct pdb_per_objfile *pdb,
		       const struct pdb_tpi_type *rec)
{
  if (rec->data_len < 6)
    {
      pdb_warning ("LF_BITFIELD record truncated (got %u)", rec->data_len);
      return pdb_tpi_get_unsupported_type (pdb);
    }

  auto base_ti = UINT32_CAST (rec->data);

  return pdb_tpi_resolve_type_internal (pdb, base_ti);
}

/* Resolve argument list from LF_ARGLIST record and populate func_type fields.  */
static void
pdb_tpi_resolve_arglist (struct pdb_per_objfile *pdb,
			 struct type *func_type,
			 uint32_t arglist_ti)
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

  if (rec->data_len < 4)
    {
      pdb_warning ("LF_ARGLIST: type index 0x%x truncated (data_len=%u)",
		   arglist_ti, rec->data_len);
      return;
    }

  auto argc = UINT32_CAST (rec->data);
  if (argc == 0 || rec->data_len < 4 + argc * 4)
    return;

  func_type->alloc_fields (argc);
  func_type->set_num_fields (argc);
  for (uint32_t i = 0; i < argc; i++)
    {
      auto arg_ti = UINT32_CAST (rec->data + 4 + i * 4);
      auto *arg_type = pdb_tpi_resolve_type_internal (pdb, arg_ti);
      func_type->field (i).set_type (arg_type);
    }
}

/* LF_PROCEDURE (0x1008).  */
static struct type *
pdb_tpi_make_procedure (struct pdb_per_objfile *pdb,
			const struct pdb_tpi_type *rec)
{
  if (rec->data_len < LF_PROC_SIZE)
    {
      pdb_warning ("LF_PROCEDURE record truncated (got %u)", rec->data_len);
      return pdb_tpi_get_unsupported_type (pdb);
    }

  auto ret_ti = UINT32_CAST (rec->data + LF_PROC_RVTYPE_OFFS);
  auto arglist_ti = UINT32_CAST (rec->data + LF_PROC_ARGLIST_OFFS);

  auto *ret_type = pdb_tpi_resolve_type_internal (pdb, ret_ti);
  auto *func_type = lookup_function_type (ret_type);

  pdb_tpi_resolve_arglist (pdb, func_type, arglist_ti);
  return func_type;
}

/* LF_MFUNCTION (0x1009) — Member function.
   Layout: ret_ti, class_ti, this_ti, ..., arglist_ti
   The class_ti and this_ti are currently unused but reserved for future.  */
static struct type *
pdb_tpi_make_mfunction (struct pdb_per_objfile *pdb,
			const struct pdb_tpi_type *rec)
{
  if (rec->data_len < LF_MFUNC_SIZE)
    {
      pdb_warning ("LF_MFUNCTION record truncated (got %u)", rec->data_len);
      return pdb_tpi_get_unsupported_type (pdb);
    }

  auto ret_ti = UINT32_CAST (rec->data + LF_MFUNC_RVTYPE_OFFS);
  auto arglist_ti = UINT32_CAST (rec->data + LF_MFUNC_ARGLIST_OFFS);

  auto *ret_type = pdb_tpi_resolve_type_internal (pdb, ret_ti);
  auto *func_type = lookup_function_type (ret_type);

  pdb_tpi_resolve_arglist (pdb, func_type, arglist_ti);
  return func_type;
}

/* Internal recursive type resolver with caching.
   Simple types are resolved inline; compound types dispatch to
   pdb_tpi_make_* helpers.  Unsupported leaf types return TYPE_CODE_ERROR.  */
static struct type *
pdb_tpi_resolve_type_internal (struct pdb_per_objfile *pdb, uint32_t type_idx)
{
  /* Simple / built-in types.  */
  if (CV_TI_IS_SIMPLE (type_idx))
    return pdb_tpi_resolve_simple_type (pdb, type_idx);

  /* Return cached type if available.  */
  if (pdb->tpi.type_cache != nullptr && pdb->tpi.type_cache[type_idx] != nullptr)
    return pdb->tpi.type_cache[type_idx];

  /* Look up compound type record from TPI stream.  */
  const struct pdb_tpi_type *rec = pdb_tpi_get_type (&pdb->tpi, type_idx);
  if (rec == nullptr)
    return pdb_tpi_get_unsupported_type (pdb);

  struct type *result = nullptr;

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
    case LF_ARGLIST:
    case LF_FIELDLIST:
    case LF_VTSHAPE:
    case LF_LABEL:
    case LF_METHODLIST:
      result = builtin_type (pdb->objfile->arch ())->builtin_void;
      break;
    default:
      /* Unsupported leaf type — return UNDEFINED.  */
      break;
    }

  if (result == nullptr)
    result = pdb_tpi_get_unsupported_type (pdb);

  /* Cache the result.  */
  if (pdb->tpi.type_cache != nullptr)
    pdb->tpi.type_cache[type_idx] = result;

  return result;
}

/* Public entry point: resolve type index to GDB type.  */
struct type *
pdb_tpi_resolve_type (struct pdb_per_objfile *pdb, uint32_t type_idx)
{
  return pdb_tpi_resolve_type_internal (pdb, type_idx);
}

/* See pdb.h.  */
int
pdb_tpi_get_func_param_count (struct pdb_per_objfile *pdb, uint32_t type_idx)
{
  const struct pdb_tpi_type *rec = pdb_tpi_get_type (&pdb->tpi, type_idx);
  if (rec == nullptr)
    return 0;

  if (rec->leaf == LF_MFUNCTION && rec->data_len >= LF_MFUNC_SIZE)
    {
      /* Add 1 for implicit 'this' parameter.  */
      return UINT16_CAST (rec->data + LF_MFUNC_PARMCOUNT_OFFS) + 1;
    }
  else if (rec->leaf == LF_PROCEDURE && rec->data_len >= LF_PROC_SIZE)
    {
      return UINT16_CAST (rec->data + LF_PROC_PARMCOUNT_OFFS);
    }

  return 0;
}