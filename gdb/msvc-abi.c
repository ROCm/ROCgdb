/* MSVC C++ ABI support for GDB.

   Copyright (C) 2026 Free Software Foundation, Inc.

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

/* MSVC C++ ABI support for GDB -- cp_abi_ops for the Microsoft Visual
   C++ ABI used by MSVC and clang-MSVC targets.  */

#include "cp-abi.h"
#include "gdbtypes.h"
#include "language.h"
#include "value.h"
#include "dwarf2.h"
#include "dwarf2/loc.h"
#include "dwarf2/expr.h"
#include "objfiles.h"
#include "extract-store-integer.h"
#include "observable.h"
#include "osabi.h"
#include "valprint.h"
#include "gdbcore.h"

/* MSVC constructor mangled names begin with "??0".  */

static enum ctor_kinds
msvc_is_constructor_name (const char *name)
{
  if (name[0] == '?' && name[1] == '?' && name[2] == '0')
    return complete_object_ctor;
  return (enum ctor_kinds) 0;
}

/* MSVC destructor mangled names begin with "??1".  */

static enum dtor_kinds
msvc_is_destructor_name (const char *name)
{
  if (name[0] == '?' && name[1] == '?' && name[2] == '1')
    return complete_object_dtor;
  return (enum dtor_kinds) 0;
}

/* MSVC vtable symbols begin with "??_7".  */

static int
msvc_is_vtable_name (const char *name)
{
  return (name[0] == '?' && name[1] == '?' && name[2] == '_'
	  && name[3] == '7');
}

/* Any "??" prefix not matching ctor/dtor/vtable is treated as an operator.
   Used for display only, so false positives are harmless.  */

static int
msvc_is_operator_name (const char *name)
{
  return (name[0] == '?' && name[1] == '?'
	  && name[2] != '0'   /* not ctor */
	  && name[2] != '1'   /* not dtor */
	  && !(name[2] == '_' && name[3] == '7')); /* not vtable */
}

/* Return pass-by-reference info for TYPE under MSVC x64 rules.

   The MSVC x64 ABI passes structs whose size is not 1, 2, 4, or 8 bytes
   via a hidden pointer regardless of triviality.  For register-sized
   structs, triviality is determined by gnuv3_pass_by_reference.  */

/* gnuv3_pass_by_reference and gnuv3_get_virtual_fn are declared in
   cp-abi.h; made non-static in gnu-v3-abi.c for reuse here.  */

static struct language_pass_by_ref_info
msvc_pass_by_reference (struct type *type)
{
  type = check_typedef (type);

  if (type->code () != TYPE_CODE_STRUCT
      && type->code () != TYPE_CODE_UNION)
    {
      struct language_pass_by_ref_info info;
      return info;
    }

  /* Delegate triviality analysis to gnu-v3, then override for
     non-register-sized structs: MSVC requires a hidden pointer for any
     struct whose size is not 1, 2, 4, or 8 bytes.  trivially_copyable
     is cleared to trigger the sret path; trivially_copy_constructible
     is left alone so infcall uses bitwise copy, not a copy constructor.  */
  struct language_pass_by_ref_info info = gnuv3_pass_by_reference (type);
  ULONGEST len = type->length ();
  if (len != 1 && len != 2 && len != 4 && len != 8)
    {
      info.trivially_copyable = false;
    }

  return info;
}

/* Return true if BATON matches the canonical clang-MSVC vbase DWARF
   expression (dup/deref/constu N/minus/deref/plus) and set *SLOT_OFFSET
   to N.  The expression encodes a vbtable lookup; the minus direction is
   a 32-bit artifact, the real offset is vbptr + N.  */

static bool
msvc_match_vbase_expr (const dwarf2_locexpr_baton *baton,
		       ULONGEST *slot_offset)
{
  gdb::array_view<const gdb_byte> expr = baton->expr ();
  const gdb_byte *p = expr.data ();
  const gdb_byte *end = p + expr.size ();

  if (p >= end || *p++ != DW_OP_dup)
    return false;
  if (p >= end || *p++ != DW_OP_deref)
    return false;
  if (p >= end || *p++ != DW_OP_constu)
    return false;
  uint64_t uleb_val;
  p = gdb_read_uleb128 (p, end, &uleb_val);
  if (p == nullptr)
    return false;
  *slot_offset = (ULONGEST) uleb_val;
  if (p >= end || *p++ != DW_OP_minus)
    return false;
  if (p >= end || *p++ != DW_OP_deref)
    return false;
  if (p >= end || *p++ != DW_OP_plus)
    return false;
  return (p == end);
}

/* Return the byte offset of base class INDEX within TYPE.

   For non-virtual bases the static DWARF bitpos is used.  For virtual
   bases clang-MSVC emits a DWARF block expression; we pattern-match the
   canonical form and read the 4-byte vbtable entry directly.  For
   non-canonical expressions the generic DWARF evaluator is used.  */

static int
msvc_baseclass_offset (struct type *type, int index,
		       const bfd_byte *valaddr, LONGEST embedded_offset,
		       CORE_ADDR address, const struct value *val)
{
  /* Non-virtual bases: the static bitpos in DWARF is correct for MSVC too.  */
  if (!BASETYPE_VIA_VIRTUAL (type, index))
    return TYPE_BASECLASS_BITPOS (type, index) / 8;

  /* Virtual base: try the canonical clang-MSVC vbtable expression first.  */
  if (type->field (index).loc_kind () == FIELD_LOC_KIND_DWARF_BLOCK_ADDR)
    {
      const dwarf2_locexpr_baton *baton
	= type->field (index).loc_dwarf_block ();
      ULONGEST slot_offset;

      if (msvc_match_vbase_expr (baton, &slot_offset))
	{
	  /* Read the vbptr, then the 32-bit signed vbtable entry at
	     vbptr + slot_offset.  The DWARF uses minus but the real
	     direction is positive; entries are 4-byte, not addr_size.  */
	  CORE_ADDR object_addr = address + embedded_offset;
	  struct gdbarch *gdbarch = type->arch ();
	  enum bfd_endian byte_order = gdbarch_byte_order (gdbarch);
	  int ptr_len
	    = builtin_type (gdbarch)->builtin_data_ptr->length ();

	  CORE_ADDR vbptr
	    = read_memory_unsigned_integer (object_addr, ptr_len, byte_order);

	  LONGEST vbase_disp
	    = read_memory_integer (vbptr + slot_offset, 4, byte_order);

	  return (int) vbase_disp;
	}

      /* Non-canonical: fall back to the generic DWARF evaluator.  */
      struct dwarf2_property_baton pbaton;
      pbaton.property_type
	= lookup_pointer_type (type->field (index).type ());
      pbaton.locexpr = *baton;

      struct dynamic_prop prop;
      prop.set_locexpr (&pbaton);

      struct property_addr_info addr_stack;
      addr_stack.type = type;
      addr_stack.addr = address + embedded_offset;
      addr_stack.next = nullptr;

      CORE_ADDR result;
      if (dwarf2_evaluate_property (&prop, nullptr, &addr_stack, &result,
				    {addr_stack.addr}))
	return (int) (result - addr_stack.addr);
    }

  /* No DWARF expression or evaluation failed.  Use the static bitpos;
     error rather than return garbage for a missing virtual base.  */
  LONGEST bitpos = TYPE_BASECLASS_BITPOS (type, index);
  if (bitpos == 0 && TYPE_N_BASECLASSES (type) > 0)
    error (_("MSVC virtual base offset not available "
	     "(no DWARF location expression for base class %d)"), index);

  return bitpos / 8;
}

/* MSVC x64 single-inheritance method pointer: one pointer-sized slot
   holding either the function address (non-virtual) or the vtable byte
   offset with lsb set (virtual).  Multi-inheritance variants (12 or
   16 bytes) are not implemented.  */

static int
msvc_method_ptr_size (struct type *type)
{
  return builtin_type (type->arch ())->builtin_func_ptr->length ();
}

/* Encode VALUE into a method pointer of TYPE; set lsb if IS_VIRTUAL.  */

static void
msvc_make_method_ptr (struct type *type, gdb_byte *contents,
		      CORE_ADDR value, int is_virtual)
{
  int size = builtin_type (type->arch ())->builtin_func_ptr->length ();
  enum bfd_endian byte_order = type_byte_order (type);

  /* Virtual: set lsb to distinguish from non-virtual.  */
  if (is_virtual)
    value |= 1;

  store_unsigned_integer (contents, size, byte_order, value);
}

/* Print a method pointer to STREAM.  */

static void
msvc_print_method_ptr (const gdb_byte *contents,
		       struct type *type,
		       struct ui_file *stream)
{
  struct type *self_type = TYPE_SELF_TYPE (type);
  struct gdbarch *gdbarch = self_type->arch ();
  struct type *funcptr_type = builtin_type (gdbarch)->builtin_func_ptr;
  CORE_ADDR ptr_value
    = extract_typed_address (contents, funcptr_type);

  if (ptr_value == 0)
    {
      gdb_printf (stream, "NULL");
      return;
    }

  int vbit = ptr_value & 1;
  CORE_ADDR func_addr = ptr_value & ~(CORE_ADDR) 1;

  if (vbit)
    {
      /* Virtual: lsb-cleared value is vtable byte offset.  */
      gdb_printf (stream, "&virtual table offset %s",
		  plongest (func_addr));
    }
  else
    {
      struct value_print_options opts;
      get_user_print_options (&opts);
      print_address_demangle (&opts, gdbarch, func_addr, stream, demangle);
    }
}

/* Resolve a method pointer to a callable value; adjusts THIS_P.  */

static struct value *
msvc_method_ptr_to_value (struct value **this_p, struct value *method_ptr)
{
  struct gdbarch *gdbarch;
  const gdb_byte *contents = method_ptr->contents ().data ();
  struct type *self_type
    = TYPE_SELF_TYPE (check_typedef (method_ptr->type ()));
  struct type *method_type
    = check_typedef (method_ptr->type ())->target_type ();
  struct type *final_type = lookup_pointer_type (self_type);

  gdbarch = self_type->arch ();
  struct type *funcptr_type = builtin_type (gdbarch)->builtin_func_ptr;
  CORE_ADDR ptr_value = extract_typed_address (contents, funcptr_type);
  int vbit = ptr_value & 1;
  CORE_ADDR func_addr = ptr_value & ~(CORE_ADDR) 1;

  *this_p = value_cast (final_type, *this_p);

  if (vbit)
    {
      /* Virtual: convert byte offset to slot index and dispatch.  */
      LONGEST voffset = func_addr / funcptr_type->length ();
      return gnuv3_get_virtual_fn (gdbarch, value_ind (*this_p),
				   method_type, voffset);
    }
  else
    return value_from_pointer (lookup_pointer_type (method_type), func_addr);
}

static struct cp_abi_ops msvc_abi_ops;

/* Populate the msvc_abi_ops vtable.  */

static void
init_msvc_abi_ops (void)
{
  msvc_abi_ops.shortname = "msvc";
  msvc_abi_ops.longname  = "Microsoft Visual C++ ABI";
  msvc_abi_ops.doc       = "MSVC C++ ABI (x64): struct-by-pointer passing "
			    "and vbptr-based virtual base offsets.";

  msvc_abi_ops.is_constructor_name = msvc_is_constructor_name;
  msvc_abi_ops.is_destructor_name  = msvc_is_destructor_name;
  msvc_abi_ops.is_vtable_name      = msvc_is_vtable_name;
  msvc_abi_ops.is_operator_name    = msvc_is_operator_name;

  msvc_abi_ops.pass_by_reference   = msvc_pass_by_reference;
  msvc_abi_ops.baseclass_offset    = msvc_baseclass_offset;

  msvc_abi_ops.method_ptr_size     = msvc_method_ptr_size;
  msvc_abi_ops.make_method_ptr     = msvc_make_method_ptr;
  msvc_abi_ops.print_method_ptr    = msvc_print_method_ptr;
  msvc_abi_ops.method_ptr_to_value = msvc_method_ptr_to_value;

  /* Remaining ops (virtual_fn_field, rtti_type, print_vtable,
     get_typeid, skip_trampoline, etc.) are left NULL -- follow-up
     work; cp-abi.c returns "not supported" for NULL entries.  */
}

/* Select the MSVC or gnu-v3 ABI based on the main executable's mangled
   symbol prefixes: "??" indicates MSVC, "_Z" indicates Itanium.  Only
   the main executable is scanned; system DLLs always contain "??"-prefixed
   symbols and would otherwise trigger this on gcc-mingw binaries.  */

static void
msvc_abi_new_objfile (struct objfile &objfile)
{
  /* Windows targets only.  */
  if (gdbarch_osabi (objfile.arch ()) != GDB_OSABI_WINDOWS)
    return;

  /* Main executable only -- system DLLs always have ??-prefixed symbols
     and would incorrectly trigger MSVC ABI selection for gcc-mingw
     binaries.  */
  if ((objfile.flags & OBJF_MAINLINE) == 0)
    return;

  /* Scan minsyms: "??" prefix means MSVC mangling, "_Z" means Itanium.
     DW_AT_producer is not available at new_objfile time.  */

  bool has_msvc_sym = false;
  bool has_itanium_sym = false;

  for (minimal_symbol *msym : objfile.msymbols ())
    {
      const char *name = msym->linkage_name ();
      if (name == nullptr)
	continue;
      if (name[0] == '?' && name[1] == '?')
	{
	  has_msvc_sym = true;
	  break;
	}
      if (name[0] == '_' && name[1] == 'Z')
	has_itanium_sym = true;
    }

  if (has_msvc_sym)
    set_cp_abi_as_auto_default ("msvc");
  else if (has_itanium_sym)
    set_cp_abi_as_auto_default ("gnu-v3");
  /* else: no C++ symbols -- leave ABI unchanged.  */
}

INIT_GDB_FILE (msvc_abi)
{
  init_msvc_abi_ops ();
  register_cp_abi (&msvc_abi_ops);
  gdb::observers::new_objfile.attach (msvc_abi_new_objfile, "msvc-abi");
}
