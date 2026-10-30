/* C++ ABI support for the Microsoft Visual C++ ABI, used by MSVC and by
   Clang targeting *-windows-msvc.

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

#include "cp-abi.h"
#include "cp-support.h"
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

/* "??0" constructor, "??$?0" template constructor.  */

static enum ctor_kinds
msvc_is_constructor_name (const char *name)
{
  if (startswith (name, "??0") || startswith (name, "??$?0"))
    return complete_object_ctor;
  return (enum ctor_kinds) 0;
}

/* "??1" destructor, "??_D" vbase destructor (complete object),
   "??_G" / "??_E" scalar / vector deleting destructor.  "??1" is the
   base object destructor when the class has virtual bases; callers
   only test for non-zero.  */

static enum dtor_kinds
msvc_is_destructor_name (const char *name)
{
  if (startswith (name, "??1"))
    return complete_object_dtor;
  if (startswith (name, "??_G") || startswith (name, "??_E"))
    return deleting_dtor;
  if (startswith (name, "??_D"))
    return complete_object_dtor;
  return (enum dtor_kinds) 0;
}

/* MSVC vtable symbols begin with "??_7".  */

static int
msvc_is_vtable_name (const char *name)
{
  return startswith (name, "??_7");
}

/* NAME is source text, not a mangled name; same as gnu-v3.  */

static int
msvc_is_operator_name (const char *name)
{
  return startswith (name, CP_OPERATOR_STR);
}

/* Return pass-by-reference info for TYPE under MSVC x64 rules.

   The MSVC x64 ABI passes structs whose size is not 1, 2, 4, or 8 bytes
   via a hidden pointer regardless of triviality.  For register-sized
   structs, triviality is determined by gnuv3_pass_by_reference.  */

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
    info.trivially_copyable = false;

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
	  /* See msvc_match_vbase_expr.  vbtable entries are 32-bit.

	     FIXME: this assumes the vbptr is at offset 0 of the object.
	     When the class also has its own vfptr, MSVC puts the vfptr
	     there and the vbptr after it, and this reads the vfptr
	     instead.  The DWARF does not give the vbptr's offset.  */
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

  /* The offset of a virtual base depends on the complete object, so a
     static offset cannot be trusted, even when it is zero.  */
  error (_("Cannot find the offset of virtual base class %d of %s"),
	 index, type->name () != nullptr ? type->name () : "<anonymous>");
}

/* Inheritance models of the MSVC C++ ABI.  They determine the size and
   layout of pointers to members of a class.  */

enum msvc_inheritance_model
{
  /* No bases, or a chain of single non-virtual bases.  */
  MSVC_INHERITANCE_SINGLE,

  /* Several bases, or a polymorphic class with a non-polymorphic base.  */
  MSVC_INHERITANCE_MULTIPLE,

  /* At least one virtual base, directly or indirectly.  */
  MSVC_INHERITANCE_VIRTUAL,

  /* The class is not complete in the debug info.  */
  MSVC_INHERITANCE_UNKNOWN,
};

/* Return true if TYPE, or one of its bases, has a virtual base.  */

static bool
msvc_has_virtual_base (struct type *type)
{
  type = check_typedef (type);
  for (int i = 0; i < TYPE_N_BASECLASSES (type); i++)
    if (BASETYPE_VIA_VIRTUAL (type, i)
	|| msvc_has_virtual_base (type->field (i).type ()))
      return true;
  return false;
}

/* Return true if TYPE has a vfptr: it, or one of its bases, declares a
   virtual function.  */

static bool
msvc_is_polymorphic (struct type *type)
{
  type = check_typedef (type);
  for (int i = 0; i < TYPE_NFN_FIELDS (type); i++)
    {
      struct fn_field *f = TYPE_FN_FIELDLIST1 (type, i);
      for (int j = 0; j < TYPE_FN_FIELDLIST_LENGTH (type, i); j++)
	if (TYPE_FN_FIELD_VIRTUAL_P (f, j))
	  return true;
    }
  for (int i = 0; i < TYPE_N_BASECLASSES (type); i++)
    if (msvc_is_polymorphic (type->field (i).type ()))
      return true;
  return false;
}

/* Return the inheritance model of class TYPE, following the rules used
   by MSVC and Clang for a class that is complete where the pointer to
   member is formed.  The model can also be forced in the source
   (__single_inheritance etc., #pragma pointers_to_members); the debug
   info does not record that.  */

static enum msvc_inheritance_model
msvc_inheritance_model (struct type *type)
{
  type = check_typedef (type);
  if (type->is_stub ())
    return MSVC_INHERITANCE_UNKNOWN;

  if (msvc_has_virtual_base (type))
    return MSVC_INHERITANCE_VIRTUAL;

  while (TYPE_N_BASECLASSES (type) > 0)
    {
      if (TYPE_N_BASECLASSES (type) > 1)
	return MSVC_INHERITANCE_MULTIPLE;

      struct type *base = check_typedef (type->field (0).type ());
      if (msvc_is_polymorphic (type) && !msvc_is_polymorphic (base))
	return MSVC_INHERITANCE_MULTIPLE;
      type = base;
    }

  return MSVC_INHERITANCE_SINGLE;
}

/* Return the size of a pointer to a member function of TYPE (a method
   type).  On x64 the size depends on the inheritance model: single 8,
   multiple 16 (function, 32-bit this adjustment), virtual 16 (function,
   32-bit this adjustment, 32-bit vbtable index), and 24 for a class that
   is incomplete where the pointer type is formed.  This is called while
   the debug info is read, when the class's bases may not be known yet,
   so return the single inheritance size here and check the model when
   the pointer is used (msvc_check_single_method_ptr).  */

static int
msvc_method_ptr_size (struct type *type)
{
  return builtin_type (type->arch ())->builtin_func_ptr->length ();
}

/* Only pointers to member functions of single-inheritance classes are
   supported.  Error out for METHODPTR_TYPE otherwise, rather than read
   the pointer wrongly.

   The length of METHODPTR_TYPE is not checked: with -readnow the debug
   info is read before the C++ ABI is selected, so the type may have the
   size another ABI gave it.  A single inheritance pointer is the
   function address in the first 8 bytes in any case.  */

static void
msvc_check_single_method_ptr (struct type *methodptr_type)
{
  struct type *self_type = TYPE_SELF_TYPE (methodptr_type);
  const char *name = (check_typedef (self_type)->name () != nullptr
		      ? check_typedef (self_type)->name () : "<anonymous>");

  if (msvc_inheritance_model (self_type) != MSVC_INHERITANCE_SINGLE)
    error (_("Pointers to member functions of class %s are not supported:"
	     " only the MSVC single inheritance model is implemented"),
	   name);
}

/* Store a pointer to a member function of TYPE in CONTENTS.  VALUE is
   the function's address, or its vtable offset if IS_VIRTUAL.  */

static void
msvc_make_method_ptr (struct type *type, gdb_byte *contents,
		      CORE_ADDR value, int is_virtual)
{
  /* A null pointer is all zeros in every model.  */
  if (value == 0 && is_virtual == 0)
    {
      memset (contents, 0, type->length ());
      return;
    }

  /* A pointer to a virtual function holds the address of a vcall thunk
     emitted by the compiler; GDB cannot create one.  */
  if (is_virtual != 0)
    error (_("Cannot create a pointer to a virtual member function"
	     " under the MSVC C++ ABI"));

  msvc_check_single_method_ptr (type);
  int ptr_len = builtin_type (type->arch ())->builtin_func_ptr->length ();
  memset (contents, 0, type->length ());
  store_unsigned_integer (contents, ptr_len, type_byte_order (type), value);
}

/* Print a pointer to a member function to STREAM.  For a virtual
   function this prints the vcall thunk the pointer holds.  */

static void
msvc_print_method_ptr (const gdb_byte *contents,
		       struct type *type,
		       struct ui_file *stream)
{
  msvc_check_single_method_ptr (type);

  struct gdbarch *gdbarch = type->arch ();
  struct type *funcptr_type = builtin_type (gdbarch)->builtin_func_ptr;
  CORE_ADDR func_addr = extract_typed_address (contents, funcptr_type);

  if (func_addr == 0)
    {
      gdb_printf (stream, "NULL");
      return;
    }

  const value_print_options &opts = get_user_print_options ();
  print_address_demangle (opts, gdbarch, func_addr, stream, demangle);
}

/* Return the function a pointer to a member function refers to, and
   cast *THIS_P to the class.  For a virtual function the pointer holds
   a vcall thunk, which does the virtual dispatch when called.  */

static struct value *
msvc_method_ptr_to_value (struct value **this_p, struct value *method_ptr)
{
  struct type *methodptr_type = check_typedef (method_ptr->type ());
  msvc_check_single_method_ptr (methodptr_type);

  struct type *self_type = TYPE_SELF_TYPE (methodptr_type);
  struct type *method_type = methodptr_type->target_type ();
  struct gdbarch *gdbarch = self_type->arch ();
  struct type *funcptr_type = builtin_type (gdbarch)->builtin_func_ptr;
  CORE_ADDR func_addr
    = extract_typed_address (method_ptr->contents ().data (), funcptr_type);

  *this_p = value_cast (lookup_pointer_type (self_type), *this_p);
  return value_from_pointer (lookup_pointer_type (method_type), func_addr);
}

static struct cp_abi_ops msvc_abi_ops;

/* Populate the msvc_abi_ops vtable.  */

static void
init_msvc_abi_ops (void)
{
  msvc_abi_ops.shortname = "msvc";
  msvc_abi_ops.longname = "Microsoft Visual C++ ABI";
  msvc_abi_ops.doc = "MSVC C++ ABI (x64).";

  msvc_abi_ops.is_constructor_name = msvc_is_constructor_name;
  msvc_abi_ops.is_destructor_name = msvc_is_destructor_name;
  msvc_abi_ops.is_vtable_name = msvc_is_vtable_name;
  msvc_abi_ops.is_operator_name = msvc_is_operator_name;

  msvc_abi_ops.pass_by_reference = msvc_pass_by_reference;
  msvc_abi_ops.baseclass_offset = msvc_baseclass_offset;

  msvc_abi_ops.method_ptr_size = msvc_method_ptr_size;
  msvc_abi_ops.make_method_ptr = msvc_make_method_ptr;
  msvc_abi_ops.print_method_ptr = msvc_print_method_ptr;
  msvc_abi_ops.method_ptr_to_value = msvc_method_ptr_to_value;

  /* Remaining ops (virtual_fn_field, rtti_type, print_vtable,
     get_typeid, skip_trampoline, etc.) are left NULL -- follow-up
     work; cp-abi.c returns "not supported" for NULL entries.  */
}

/* Select the C++ ABI from the main executable's mangled names, or
   from the OS ABI if it has no C++ symbols.  The OS ABI alone is not
   enough: it comes from the configured target, not from the binary.  */

static void
msvc_abi_new_objfile (struct objfile &objfile)
{
  /* Windows targets only.  */
  enum gdb_osabi osabi = gdbarch_osabi (objfile.arch ());
  if (osabi != GDB_OSABI_WINDOWS_GNU && osabi != GDB_OSABI_WINDOWS_MSVC)
    return;

  /* Main executable only; system DLLs always have "??" symbols.  */
  if ((objfile.flags & OBJF_MAINLINE) == 0)
    return;

  /* "??" means MSVC mangling, "_Z" means Itanium.  */

  bool has_msvc_sym = false;
  bool has_itanium_sym = false;

  for (minimal_symbol *msym : objfile.msymbols ())
    {
      const char *name = msym->linkage_name ();
      if (name == nullptr)
	continue;
      if (startswith (name, "??"))
	{
	  has_msvc_sym = true;
	  break;
	}
      if (startswith (name, "_Z"))
	has_itanium_sym = true;
    }

  if (has_msvc_sym)
    set_cp_abi_as_auto_default ("msvc");
  else if (has_itanium_sym)
    set_cp_abi_as_auto_default ("gnu-v3");
  else if (osabi == GDB_OSABI_WINDOWS_MSVC)
    set_cp_abi_as_auto_default ("msvc");
  else
    set_cp_abi_as_auto_default ("gnu-v3");
}

INIT_GDB_FILE (msvc_abi)
{
  init_msvc_abi_ops ();
  register_cp_abi (&msvc_abi_ops);
  gdb::observers::new_objfile.attach (msvc_abi_new_objfile, "msvc-abi");
}
