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
#include "gdb-demangle.h"
#include "expression.h"
#include "minsyms.h"
#include "target.h"
#include <algorithm>

/* "??0" constructor, "??$?0" template constructor.

   FIXME: The name checks in this file use raw MSVC mangling prefixes.
   Once GDB has an MSVC demangler, recognise these special members from
   the demangled name instead, which also covers forms not listed
   here.  */

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

static bool msvc_has_virtual_base (struct type *type);

/* Return the offset of the vbptr of class TYPE in an object of TYPE at
   ADDR.  A class shares the vbptr of its first non-virtual base, in
   declaration order, that has virtual bases; otherwise it has its own,
   usually after its own vfptr.  Find that one by scanning: the first
   entry of its vbtable is minus its offset.  */

static LONGEST
msvc_find_vbptr_offset (struct type *type, CORE_ADDR addr)
{
  type = check_typedef (type);
  for (int i = 0; i < TYPE_N_BASECLASSES (type); i++)
    {
      struct type *base = check_typedef (type->field (i).type ());
      if (!BASETYPE_VIA_VIRTUAL (type, i) && msvc_has_virtual_base (base))
	{
	  LONGEST off = TYPE_BASECLASS_BITPOS (type, i) / 8;
	  return off + msvc_find_vbptr_offset (base, addr + off);
	}
    }

  struct gdbarch *gdbarch = type->arch ();
  enum bfd_endian byte_order = gdbarch_byte_order (gdbarch);
  int ptr_len = builtin_type (gdbarch)->builtin_data_ptr->length ();
  LONGEST found = -1;
  for (LONGEST s = 0; s + ptr_len <= type->length (); s += ptr_len)
    {
      CORE_ADDR vbtable;
      try
	{
	  vbtable = read_memory_unsigned_integer (addr + s, ptr_len,
						  byte_order);
	  if (read_memory_integer (vbtable, 4, byte_order) != -s)
	    continue;
	}
      catch (const gdb_exception_error &)
	{
	  continue;
	}

      /* Prefer a vbtable that has a "??_8" symbol; without symbols,
	 take the first match.  */
      bound_minimal_symbol bmsym = lookup_minimal_symbol_by_pc (vbtable);
      if (bmsym.minsym != nullptr
	  && startswith (bmsym.minsym->linkage_name (), "??_8"))
	return s;
      if (found < 0)
	found = s;
    }

  if (found < 0)
    error (_("Cannot find the vbtable pointer in %s"),
	   type->name () != nullptr ? type->name () : "<anonymous>");
  return found;
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

	     The DWARF expression assumes the vbptr is at offset 0 of the
	     subobject, but MSVC places the class's own vfptr first and the
	     vbptr after it.  Scan pointer-aligned slots to find the vbptr.
	     A slot S holds the vbptr when the vbtable it points to has
	     entry[0] == -S (entries are signed offsets from the vbptr address,
	     so entry[0] records the negated vbptr offset within the object).
	     Prefer a slot whose vbtable address matches a "??_8" minsym.  */
	  CORE_ADDR object_addr = address + embedded_offset;
	  struct gdbarch *gdbarch = type->arch ();
	  enum bfd_endian byte_order = gdbarch_byte_order (gdbarch);
	  int ptr_len
	    = builtin_type (gdbarch)->builtin_data_ptr->length ();
	  LONGEST vbptr_off = msvc_find_vbptr_offset (type, object_addr);

	  CORE_ADDR vbtable
	    = read_memory_unsigned_integer (object_addr + vbptr_off,
					    ptr_len, byte_order);
	  LONGEST vbase_disp
	    = read_memory_integer (vbtable + slot_offset, 4, byte_order);
	  return (int) (vbptr_off + vbase_disp);
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
   the pointer is used (msvc_check_single_method_ptr).

   FIXME: sizeof gives 8 for multiple and virtual inheritance pointers,
   which are 16 bytes in the inferior.  Fixing this needs the model to
   be decided after the class is complete, e.g. when the pointer type
   is resolved by check_typedef.  */

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

  /* FIXME: For a virtual function this prints the raw vcall thunk
     symbol (e.g. "??_9S@@$B7AA").  Gnu-v3 prints "&virtual S::g()";
     doing the same needs the vtable slot from the thunk (the
     "$B<offset>" part of its name, or a decode of its code, see
     amd64-windows-tdep.c) mapped back to the virtual function.  Once
     GDB has an MSVC demangler, at least print the demangled thunk
     name.  */
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

/* Forward declaration for msvc_find_vfptr (defined in the RTTI section).  */
static CORE_ADDR msvc_find_vfptr (struct type *type, CORE_ADDR addr);

/* Virtual dispatch.  */

/* Return true if any fn_fieldlist of TYPE declares a virtual function
   whose bare name matches NAME.  */

static bool
msvc_type_has_virtual_named (struct type *type, const char *name)
{
  type = check_typedef (type);
  for (int i = 0; i < TYPE_NFN_FIELDS (type); i++)
    if (strcmp (TYPE_FN_FIELDLIST_NAME (type, i), name) == 0)
      {
	struct fn_field *fl = TYPE_FN_FIELDLIST1 (type, i);
	int len = TYPE_FN_FIELDLIST_LENGTH (type, i);
	for (int j = 0; j < len; j++)
	  if (TYPE_FN_FIELD_VIRTUAL_P (fl, j))
	    return true;
      }
  return false;
}

/* Find the base subobject that first introduced the virtual function
   named NAME into the hierarchy rooted at TYPE.  Set *BASE_TYPE to
   that base's type and *BASE_ADDR to its address.  Non-virtual bases
   are tried before virtual bases, matching MSVC's layout order
   (primary base first).  */

static void
msvc_find_introducing_base (struct type *type, CORE_ADDR addr,
			    const char *name,
			    struct type **base_type, CORE_ADDR *base_addr)
{
  type = check_typedef (type);

  /* Non-virtual bases first.  */
  for (int i = 0; i < TYPE_N_BASECLASSES (type); i++)
    if (!BASETYPE_VIA_VIRTUAL (type, i))
      {
	struct type *bt = check_typedef (type->field (i).type ());
	CORE_ADDR ba = addr + TYPE_BASECLASS_BITPOS (type, i) / 8;
	if (msvc_type_has_virtual_named (bt, name))
	  {
	    msvc_find_introducing_base (bt, ba, name,
					base_type, base_addr);
	    return;
	  }
      }

  /* Virtual bases.  */
  for (int i = 0; i < TYPE_N_BASECLASSES (type); i++)
    if (BASETYPE_VIA_VIRTUAL (type, i))
      {
	struct type *bt = check_typedef (type->field (i).type ());
	if (msvc_type_has_virtual_named (bt, name))
	  {
	    int off = msvc_baseclass_offset (type, i, nullptr, 0, addr,
					     nullptr);
	    msvc_find_introducing_base (bt, addr + off, name,
					base_type, base_addr);
	    return;
	  }
      }

  /* No base declares NAME as virtual: TYPE introduces it.  */
  *base_type = type;
  *base_addr = addr;
}

/* Implement the "virtual_fn_field" method.
   Locate the vftable slot for method F[J] in *VALUE_P, set *VALUE_P
   to the introducing subobject (which infcall uses as "this"), and
   return the function address as a value.  */

static struct value *
msvc_virtual_fn_field (struct value **value_p,
		       struct fn_field *f, int j,
		       struct type *type, int offset)
{
  /* Step 1: find the introducing base subobject.
     Walk the fn_fieldlists to recover the bare method name from F.  */
  const char *name = nullptr;
  for (int i = 0; i < TYPE_NFN_FIELDS (type); i++)
    if (TYPE_FN_FIELDLIST1 (type, i) == f)
      {
	name = TYPE_FN_FIELDLIST_NAME (type, i);
	break;
      }
  if (name == nullptr)
    error (_("Cannot find virtual function name in type %s"),
	   type->name () != nullptr ? type->name () : "<anonymous>");

  /* Refuse to dispatch virtual destructors: slot 0 holds the scalar
     deleting destructor (??_G), which frees the object when called.
     Detect via the fieldlist name (starts with '~'), since the physname
     under MSVC is the demangled form, not the raw mangled one.  */
  if (name[0] == '~')
    error (_("Cannot call a virtual destructor through the vftable"
	     " under the MSVC C++ ABI"));

  struct type *values_type = check_typedef ((*value_p)->type ());
  if (values_type->code () != TYPE_CODE_STRUCT)
    error (_("Only classes can have virtual functions."));

  struct gdbarch *gdbarch = values_type->arch ();
  enum bfd_endian byte_order = gdbarch_byte_order (gdbarch);
  int ptr_len = builtin_type (gdbarch)->builtin_data_ptr->length ();

  CORE_ADDR addr = (*value_p)->address ()
    + (*value_p)->embedded_offset () + offset;

  struct type *intro_type;
  CORE_ADDR intro_addr;
  msvc_find_introducing_base (type, addr, name, &intro_type, &intro_addr);

  /* Step 2: get the vfptr of the introducing subobject.  */
  CORE_ADDR vfptr_addr = msvc_find_vfptr (intro_type, intro_addr);
  if (vfptr_addr == 0)
    vfptr_addr = intro_addr;

  /* Step 3: read the function address from the slot.  */
  ULONGEST voffset = TYPE_FN_FIELD_VOFFSET (f, j);
  CORE_ADDR vftable
    = read_memory_unsigned_integer (vfptr_addr, ptr_len, byte_order);
  CORE_ADDR fn_addr
    = read_memory_unsigned_integer (vftable + voffset * ptr_len,
				    ptr_len, byte_order);

  /* Step 4: adjust *value_p to the introducing subobject so infcall
     passes the right "this".  Create the subobject as a struct value
     at intro_addr (same shape as gnu-v3's value_cast to the base).  */
  *value_p = value_from_contents_and_address (intro_type, nullptr,
					       intro_addr);

  /* Step 5: return the function as a pointer-to-method value.  */
  return value_from_pointer (
    lookup_pointer_type (TYPE_FN_FIELD_TYPE (f, j)), fn_addr);
}

/* RTTI.

   On x64 the slot before the first virtual function of each vftable
   points to an _RTTICompleteObjectLocator (COL):

     uint32 signature;      1 on x64: the fields below are image relative
     int32  offset;         offset of this vfptr in the complete object
     int32  cdOffset;       if non-zero, a vtordisp sits cdOffset bytes
			    before the vfptr and is added to OFFSET
     uint32 pTypeDescriptor;    TypeDescriptor of the complete object
     uint32 pClassDescriptor;   class hierarchy, not used here
     uint32 pSelf;              the COL itself, gives the image base

   A TypeDescriptor is the std::type_info object of the type: a vfptr,
   a pointer to the undecorated name (filled in lazily by the runtime)
   and the decorated name, e.g. ".?AVFoo@@".  The TypeDescriptor of any
   type that is used in typeid also has a symbol, "??_R0" followed by
   the decorated name without the leading '.', followed by "@8".  */

/* A COL that has been read from the inferior.  */

struct msvc_col
{
  /* COL.offset plus the vtordisp, if any: the offset of the vfptr the
     COL was found through from the start of the complete object.  */
  LONGEST offset_to_vfptr;

  /* The address of the TypeDescriptor.  */
  CORE_ADDR type_descriptor;
};

/* Return true if NAME is the name of a vfptr field ("_vptr$Class").  */

static bool
msvc_is_vfptr_field_name (const char *name)
{
  return (name != nullptr
	  && startswith (name, "_vptr")
	  && is_cplus_marker (name[sizeof ("_vptr") - 1]));
}

/* Return the address of a vfptr in the object of class TYPE at ADDR, or
   0 if the debug info shows none.  A class only gets its own vfptr if
   no non-virtual base already has one, so look at the class's own
   fields first, then the non-virtual bases, then the virtual bases.  */

static CORE_ADDR
msvc_find_vfptr (struct type *type, CORE_ADDR addr)
{
  type = check_typedef (type);

  for (int i = TYPE_N_BASECLASSES (type); i < type->num_fields (); i++)
    if (!type->field (i).is_static ()
	&& type->field (i).loc_kind () == FIELD_LOC_KIND_BITPOS
	&& msvc_is_vfptr_field_name (type->field (i).name ()))
      return addr + type->field (i).loc_bitpos () / 8;

  for (int i = 0; i < TYPE_N_BASECLASSES (type); i++)
    if (!BASETYPE_VIA_VIRTUAL (type, i))
      {
	CORE_ADDR vfptr
	  = msvc_find_vfptr (type->field (i).type (),
			     addr + TYPE_BASECLASS_BITPOS (type, i) / 8);
	if (vfptr != 0)
	  return vfptr;
      }

  /* Only reached when the class has no vfptr outside its virtual bases.
     The offset of the virtual base comes from msvc_baseclass_offset.  */
  for (int i = 0; i < TYPE_N_BASECLASSES (type); i++)
    if (BASETYPE_VIA_VIRTUAL (type, i)
	&& msvc_is_polymorphic (type->field (i).type ()))
      {
	int offset = msvc_baseclass_offset (type, i, nullptr, 0, addr,
					    nullptr);
	CORE_ADDR vfptr
	  = msvc_find_vfptr (type->field (i).type (), addr + offset);
	if (vfptr != 0)
	  return vfptr;
      }

  return 0;
}

/* Read the COL for the polymorphic object of class TYPE at ADDR into
   *COL, and the address of the vfptr it was found through into
   *VFPTR_ADDR.  Return false if the object has no x64 COL.  Errors out
   if memory cannot be read.  */

static bool
msvc_read_object_col (struct type *type, CORE_ADDR addr,
		      CORE_ADDR *vfptr_addr, struct msvc_col *col)
{
  struct gdbarch *gdbarch = type->arch ();
  enum bfd_endian byte_order = gdbarch_byte_order (gdbarch);
  int ptr_len = builtin_type (gdbarch)->builtin_data_ptr->length ();

  /* Without a vfptr field in the debug info, assume the vfptr is at the
     start of the object, where MSVC puts the vfptr of a class that
     declares the first virtual function.  */
  *vfptr_addr = msvc_find_vfptr (type, addr);
  if (*vfptr_addr == 0)
    *vfptr_addr = addr;

  CORE_ADDR vftable
    = read_memory_unsigned_integer (*vfptr_addr, ptr_len, byte_order);
  CORE_ADDR col_addr
    = read_memory_unsigned_integer (vftable - ptr_len, ptr_len, byte_order);

  /* FIXME: Signature 0 is the 32-bit layout, with absolute pointers
     instead of image relative offsets and no pSelf.  Only x64 is
     handled.  */
  if (read_memory_unsigned_integer (col_addr, 4, byte_order) != 1)
    return false;

  LONGEST offset = read_memory_integer (col_addr + 4, 4, byte_order);
  LONGEST cd_offset = read_memory_integer (col_addr + 8, 4, byte_order);
  ULONGEST type_descriptor_rva
    = read_memory_unsigned_integer (col_addr + 12, 4, byte_order);
  ULONGEST self_rva
    = read_memory_unsigned_integer (col_addr + 20, 4, byte_order);

  /* While a constructor or destructor of a class with virtual bases
     runs, the virtual base can be at a different offset than in the
     complete object; the vtordisp before the vfptr holds the
     difference.  */
  LONGEST vtordisp = 0;
  if (cd_offset != 0)
    vtordisp = read_memory_integer (*vfptr_addr - cd_offset, 4,
				    byte_order);

  CORE_ADDR image_base = col_addr - self_rva;
  col->offset_to_vfptr = offset + vtordisp;
  col->type_descriptor = image_base + type_descriptor_rva;
  return true;
}

/* Return the decorated name stored in the TypeDescriptor at
   TYPE_DESCRIPTOR.  */

static std::string
msvc_read_decorated_name (struct gdbarch *gdbarch,
			  CORE_ADDR type_descriptor)
{
  int ptr_len = builtin_type (gdbarch)->builtin_data_ptr->length ();
  CORE_ADDR name_addr = type_descriptor + 2 * ptr_len;
  gdb::unique_xmalloc_ptr<char> name = target_read_string (name_addr, 4096);

  if (name == nullptr || name.get ()[0] != '.')
    error (_("Cannot read the type name of the type_info object at %s"),
	   paddress (gdbarch, type_descriptor));
  return name.get ();
}

/* Decorated names of the fundamental types.  */

static const struct
{
  const char *name;
  const char *code;
} msvc_fundamental_types[] =
{
  { "char", "D" },
  { "signed char", "C" },
  { "unsigned char", "E" },
  { "short", "F" },
  { "unsigned short", "G" },
  { "int", "H" },
  { "unsigned int", "I" },
  { "long", "J" },
  { "unsigned long", "K" },
  { "long long", "_J" },
  { "unsigned long long", "_K" },
  { "float", "M" },
  { "double", "N" },
  { "long double", "O" },
  { "bool", "_N" },
  { "wchar_t", "_W" },
  { "char16_t", "_S" },
  { "char32_t", "_U" },
  { "void", "X" },
};

/* A decoder for the decorated names in TypeDescriptors.  It handles
   classes, structs, unions and enums in plain and anonymous namespaces,
   the fundamental types and pointers to those.

   FIXME: Templates ("?$"), local classes ("?1", "??"), arrays ("$$B"),
   function and member pointers are not handled, and the decoder returns
   false for them.  Once GDB has an MSVC demangler, use it instead; it
   covers all of these.  */

struct msvc_decorated_name_parser
{
  explicit msvc_decorated_name_parser (const char *name)
    : m_p (name)
  {
  }

  /* Decode a type into OUT and return true, or return false.  */
  bool type (std::string &out);

  /* True if the whole name has been consumed.  */
  bool at_end () const
  { return *m_p == '\0'; }

private:

  bool qualified_name (std::string &out);

  const char *m_p;

  /* The first ten name fragments, referred to by the digits 0-9.  */
  std::vector<std::string> m_backrefs;
};

/* Decode a qualified name, e.g. "Foo@ns@@", into OUT.  */

bool
msvc_decorated_name_parser::qualified_name (std::string &out)
{
  std::vector<std::string> parts;

  while (true)
    {
      if (*m_p == '\0')
	return false;

      if (*m_p == '@')
	{
	  m_p++;
	  break;
	}

      if (*m_p >= '0' && *m_p <= '9')
	{
	  size_t index = *m_p++ - '0';
	  if (index >= m_backrefs.size ())
	    return false;
	  parts.push_back (m_backrefs[index]);
	  continue;
	}

      const char *at = strchr (m_p, '@');
      if (at == nullptr)
	return false;

      if (*m_p == '?')
	{
	  /* "?A0x<hash>@" is an anonymous namespace; it does not take a
	     back reference slot.  */
	  if (m_p[1] != 'A')
	    return false;
	  parts.push_back (CP_ANONYMOUS_NAMESPACE_STR);
	  m_p = at + 1;
	  continue;
	}

      std::string fragment (m_p, at - m_p);
      m_p = at + 1;
      if (m_backrefs.size () < 10)
	m_backrefs.push_back (fragment);
      parts.push_back (std::move (fragment));
    }

  if (parts.empty ())
    return false;

  /* The innermost name comes first.  */
  out.clear ();
  for (auto it = parts.rbegin (); it != parts.rend (); ++it)
    {
      if (!out.empty ())
	out += "::";
      out += *it;
    }
  return true;
}

bool
msvc_decorated_name_parser::type (std::string &out)
{
  switch (*m_p)
    {
    case 'T':
    case 'U':
    case 'V':
      m_p++;
      return qualified_name (out);

    case 'W':
      if (m_p[1] != '4')
	return false;
      m_p += 2;
      return qualified_name (out);

    case 'P':
    case 'Q':
    case 'R':
    case 'S':
      {
	/* The pointer's own qualifiers, if any, are also given by the
	   enclosing pointer's target qualifiers, so ignore them.  */
	if (m_p[1] != 'E')
	  return false;
	char target_cv = m_p[2];
	if (target_cv < 'A' || target_cv > 'D')
	  return false;
	m_p += 3;

	bool target_is_pointer = strchr ("PQRS", *m_p) != nullptr;
	std::string target;
	if (*m_p == '\0' || !type (target))
	  return false;

	bool is_const = target_cv == 'B' || target_cv == 'D';
	bool is_volatile = target_cv == 'C' || target_cv == 'D';
	if (target_is_pointer)
	  {
	    out = std::move (target);
	    if (is_const)
	      out += " const";
	    if (is_volatile)
	      out += " volatile";
	    out += (is_const || is_volatile) ? " *" : "*";
	  }
	else
	  {
	    out.clear ();
	    if (is_const)
	      out += "const ";
	    if (is_volatile)
	      out += "volatile ";
	    out += target;
	    out += " *";
	  }
	return true;
      }

    default:
      for (const auto &fundamental : msvc_fundamental_types)
	if (startswith (m_p, fundamental.code))
	  {
	    m_p += strlen (fundamental.code);
	    out = fundamental.name;
	    return true;
	  }
      return false;
    }
}

/* Decode the decorated name of a TypeDescriptor, e.g. ".?AVFoo@@" or
   ".PEAH", into OUT.  Return false if it is not understood.  */

static bool
msvc_decode_decorated_name (const char *name, std::string &out)
{
  if (*name++ != '.')
    return false;

  /* Classes, structs, unions and enums are prefixed with "?A".  */
  if (startswith (name, "?A"))
    name += 2;

  msvc_decorated_name_parser parser (name);
  return parser.type (out) && parser.at_end ();
}

/* Append the MSVC encoding of N to OUT.  */

static void
msvc_encode_number (std::string &out, ULONGEST n)
{
  if (n >= 1 && n <= 10)
    {
      out += (char) ('0' + n - 1);
      return;
    }

  std::string hex;
  do
    {
      hex.insert (hex.begin (), (char) ('A' + (n & 0xf)));
      n >>= 4;
    }
  while (n != 0);
  out += hex;
  out += '@';
}

/* Append the encoding of the qualified class name NAME to OUT, using
   and updating the back references in BACKREFS.  */

static bool
msvc_encode_qualified_name (const char *name, std::string &out,
			    std::vector<std::string> &backrefs)
{
  /* FIXME: Templates and anonymous namespaces (whose encoding has a
     hash GDB does not know) are not handled.  */
  if (name == nullptr
      || strchr (name, '<') != nullptr
      || strchr (name, '(') != nullptr)
    return false;

  std::vector<std::string> parts;
  for (const char *p = name;;)
    {
      const char *sep = strstr (p, "::");
      if (sep == nullptr)
	{
	  parts.emplace_back (p);
	  break;
	}
      parts.emplace_back (p, sep - p);
      p = sep + 2;
    }

  for (auto it = parts.rbegin (); it != parts.rend (); ++it)
    {
      if (it->empty ())
	return false;

      auto ref = std::find (backrefs.begin (), backrefs.end (), *it);
      if (ref != backrefs.end ())
	out += (char) ('0' + (ref - backrefs.begin ()));
      else
	{
	  out += *it;
	  out += '@';
	  if (backrefs.size () < 10)
	    backrefs.push_back (*it);
	}
    }
  out += '@';
  return true;
}

/* Append the decorated name of TYPE to OUT.  TOP is true for the type
   given to typeid, whose top-level qualifiers do not count.  If
   SWAP_CLASS_KEY, encode a struct as a class and vice versa.  */

static bool
msvc_encode_type (struct type *type, std::string &out,
		  std::vector<std::string> &backrefs, bool top,
		  bool swap_class_key)
{
  struct type *orig_type = type;
  type = check_typedef (type);

  switch (type->code ())
    {
    case TYPE_CODE_PTR:
      {
	struct type *target = check_typedef (type->target_type ());
	if (target->code () == TYPE_CODE_FUNC
	    || target->code () == TYPE_CODE_ARRAY)
	  return false;

	int cv = 0;
	if (!top)
	  cv = ((orig_type->is_const () || type->is_const ()) ? 1 : 0)
	    + ((orig_type->is_volatile () || type->is_volatile ()) ? 2 : 0);
	out += "PQRS"[cv];
	out += 'E';

	struct type *raw_target = type->target_type ();
	int target_cv
	  = ((raw_target->is_const () || target->is_const ()) ? 1 : 0)
	  + ((raw_target->is_volatile () || target->is_volatile ()) ? 2 : 0);
	out += "ABCD"[target_cv];
	return msvc_encode_type (raw_target, out, backrefs, false, false);
      }

    case TYPE_CODE_STRUCT:
      out += type->is_declared_class () != swap_class_key ? 'V' : 'U';
      return msvc_encode_qualified_name (type->name (), out, backrefs);

    case TYPE_CODE_UNION:
      out += 'T';
      return msvc_encode_qualified_name (type->name (), out, backrefs);

    case TYPE_CODE_ENUM:
      out += "W4";
      return msvc_encode_qualified_name (type->name (), out, backrefs);

    case TYPE_CODE_ARRAY:
      {
	/* Only typeid of an array type itself is encoded like this.  */
	if (!top)
	  return false;

	std::vector<ULONGEST> dims;
	while (type->code () == TYPE_CODE_ARRAY)
	  {
	    LONGEST low, high;
	    if (!get_array_bounds (type, &low, &high) || low != 0)
	      return false;
	    dims.push_back (high - low + 1);
	    type = check_typedef (type->target_type ());
	  }

	out += "$$BY";
	msvc_encode_number (out, dims.size ());
	for (ULONGEST dim : dims)
	  msvc_encode_number (out, dim);
	return msvc_encode_type (type, out, backrefs, false, false);
      }

    case TYPE_CODE_INT:
    case TYPE_CODE_CHAR:
    case TYPE_CODE_BOOL:
    case TYPE_CODE_FLT:
    case TYPE_CODE_VOID:
      if (type->name () == nullptr)
	return false;
      for (const auto &fundamental : msvc_fundamental_types)
	if (strcmp (type->name (), fundamental.name) == 0)
	  {
	    out += fundamental.code;
	    return true;
	  }
      return false;

    default:
      return false;
    }
}

/* Return in OUT the name of the symbol of the TypeDescriptor of TYPE,
   e.g. "??_R0?AVFoo@@@8".  Return false if TYPE cannot be encoded.  See
   msvc_encode_type for SWAP_CLASS_KEY.

   FIXME: Once GDB has an MSVC demangler, look the symbol up by its
   demangled name instead of encoding TYPE by hand.  */

static bool
msvc_type_descriptor_symbol (struct type *type, std::string &out,
			     bool swap_class_key)
{
  std::vector<std::string> backrefs;

  type = check_typedef (type);
  out = "??_R0";
  if (type->code () == TYPE_CODE_STRUCT
      || type->code () == TYPE_CODE_UNION
      || type->code () == TYPE_CODE_ENUM)
    out += "?A";
  if (!msvc_encode_type (type, out, backrefs, true, swap_class_key))
    return false;
  out += "@8";
  return true;
}

/* Implement the "rtti_type" method.  */

static struct type *
msvc_rtti_type (struct value *value, int *full_p, LONGEST *top_p,
		int *using_enc_p)
{
  struct type *values_type = check_typedef (value->type ());

  if (values_type->code () != TYPE_CODE_STRUCT
      || !msvc_is_polymorphic (values_type))
    return nullptr;

  struct gdbarch *gdbarch = values_type->arch ();
  CORE_ADDR addr = value_as_address (value_addr (value));
  CORE_ADDR vfptr_addr;
  struct msvc_col col;

  if (using_enc_p != nullptr)
    *using_enc_p = 0;

  if (!msvc_read_object_col (values_type, addr, &vfptr_addr, &col))
    return nullptr;

  /* Only class names are decoded for the dynamic type.  */
  std::string decorated = msvc_read_decorated_name (gdbarch,
						    col.type_descriptor);
  std::string class_name;
  if (!startswith (decorated, ".?A")
      || !msvc_decode_decorated_name (decorated.c_str (), class_name))
    return nullptr;

  struct type *run_time_type = cp_lookup_rtti_type (class_name.c_str (),
						    nullptr);
  if (run_time_type == nullptr)
    return nullptr;

  /* The offset of VALUE from the start of the complete object.  */
  LONGEST top = col.offset_to_vfptr - (LONGEST) (vfptr_addr - addr);

  if (full_p != nullptr)
    *full_p = (top == value->embedded_offset ()
	       && (value->enclosing_type ()->length ()
		   >= run_time_type->length ()));
  if (top_p != nullptr)
    *top_p = top;
  return run_time_type;
}

/* A gdbarch key for the type_info type, when the program's debug info
   does not have it.  */

static const registry<gdbarch>::key<struct type> msvc_type_info_gdbarch_data;

/* Return a type for std::type_info as laid out by the MSVC runtime.  */

static struct type *
msvc_build_type_info_type (struct gdbarch *arch)
{
  struct type *void_ptr_type = builtin_type (arch)->builtin_data_ptr;
  struct type *char_type = builtin_type (arch)->builtin_char;
  struct type *char_ptr_type
    = make_pointer_type (make_cv_type (1, 0, char_type));
  struct type *t
    = type_allocator (arch).new_type (TYPE_CODE_STRUCT, 0, nullptr);
  int ptr_len = void_ptr_type->length ();

  t->alloc_fields (3);

  t->field (0).set_name ("_vptr$type_info");
  t->field (0).set_type (void_ptr_type);
  t->field (0).set_loc_bitpos (0);

  t->field (1).set_name ("_UndecoratedName");
  t->field (1).set_type (char_ptr_type);
  t->field (1).set_loc_bitpos (ptr_len * TARGET_CHAR_BIT);

  /* The decorated name is a NUL-terminated string of any length.  */
  t->field (2).set_name ("_DecoratedName");
  t->field (2).set_type (lookup_array_range_type (char_type, 0, 0));
  t->field (2).set_loc_bitpos (2 * ptr_len * TARGET_CHAR_BIT);

  t->set_length (3 * ptr_len);
  t->set_name ("gdb_msvc_type_info");
  INIT_CPLUS_SPECIFIC (t);

  return t;
}

/* Implement the "get_typeid_type" method.  */

static struct type *
msvc_get_typeid_type (struct gdbarch *gdbarch)
{
  /* MSVC declares type_info in the global namespace; std::type_info is
     a using-declaration.  */
  for (const char *name : { "type_info", "std::type_info" })
    {
      struct symbol *sym = lookup_symbol (name, nullptr, SEARCH_STRUCT_DOMAIN,
					  nullptr).symbol;
      if (sym != nullptr)
	return sym->type ();
    }

  struct type *typeinfo_type = msvc_type_info_gdbarch_data.get (gdbarch);
  if (typeinfo_type == nullptr)
    {
      typeinfo_type = msvc_build_type_info_type (gdbarch);
      msvc_type_info_gdbarch_data.set (gdbarch, typeinfo_type);
    }
  return typeinfo_type;
}

/* Implement the "get_typeid" method.  */

static struct value *
msvc_get_typeid (struct value *value)
{
  /* As in gnu-v3: "typeid (type-id)" passes a not_lval value.  */
  if (value->lval () == lval_memory)
    value = coerce_ref (value);

  struct type *type = check_typedef (value->type ());
  if (type->code () == TYPE_CODE_REF)
    type = check_typedef (type->target_type ());

  /* Ignore top-level cv-qualifiers.  */
  type = make_cv_type (0, 0, type);
  struct gdbarch *gdbarch = type->arch ();

  std::string type_name = type_to_string (type);
  if (type_name.empty ())
    error (_("cannot find typeinfo for unnamed type"));

  struct type *typeinfo_type = msvc_get_typeid_type (gdbarch);

  if (type->code () == TYPE_CODE_STRUCT
      && value->lval () == lval_memory
      && msvc_is_polymorphic (type))
    {
      CORE_ADDR addr = value->address () + value->embedded_offset ();
      CORE_ADDR vfptr_addr;
      struct msvc_col col;

      if (!msvc_read_object_col (type, addr, &vfptr_addr, &col))
	error (_("cannot find typeinfo for object of type '%s'"),
	       type_name.c_str ());
      return value_at_lazy (typeinfo_type, col.type_descriptor);
    }

  /* The debug info does not reliably tell "struct" from "class", which
     are encoded differently, so try both.  */
  for (bool swap_class_key : { false, true })
    {
      std::string sym_name;
      if (!msvc_type_descriptor_symbol (type, sym_name, swap_class_key))
	break;

      bound_minimal_symbol minsym
	= lookup_minimal_symbol (current_program_space, sym_name.c_str ());
      if (minsym.minsym != nullptr)
	return value_at_lazy (typeinfo_type, minsym.value_address ());

      if (type->code () != TYPE_CODE_STRUCT)
	break;
    }

  error (_("could not find typeinfo symbol for '%s'"), type_name.c_str ());
}

/* Implement the "get_typename_from_type_info" method.  */

static std::string
msvc_get_typename_from_type_info (struct value *type_info_ptr)
{
  struct gdbarch *gdbarch = type_info_ptr->type ()->arch ();
  CORE_ADDR addr = value_as_address (type_info_ptr);
  std::string decorated = msvc_read_decorated_name (gdbarch, addr);
  std::string name;

  if (!msvc_decode_decorated_name (decorated.c_str (), name))
    error (_("Cannot decode the type name '%s' of the type_info object"
	     " at %s"),
	   decorated.c_str (), paddress (gdbarch, addr));
  return name;
}

/* Implement the "get_type_from_type_info" method.  */

static struct type *
msvc_get_type_from_type_info (struct value *type_info_ptr)
{
  std::string type_name = msvc_get_typename_from_type_info (type_info_ptr);
  expression_up expr (parse_expression (type_name.c_str ()));
  struct value *type_val = expr->evaluate_type ();
  return type_val->type ();
}

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

  msvc_abi_ops.rtti_type = msvc_rtti_type;
  msvc_abi_ops.get_typeid = msvc_get_typeid;
  msvc_abi_ops.get_typeid_type = msvc_get_typeid_type;
  msvc_abi_ops.get_type_from_type_info = msvc_get_type_from_type_info;
  msvc_abi_ops.get_typename_from_type_info
    = msvc_get_typename_from_type_info;

  msvc_abi_ops.virtual_fn_field = msvc_virtual_fn_field;

  /* Remaining ops (print_vtable, skip_trampoline) are left NULL --
     follow-up work; cp-abi.c returns "not supported" for NULL entries.
     Thunks are skipped by the gdbarch hook in amd64-windows-tdep.c.  */
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
