/* Modula 2 language support routines for GDB, the GNU debugger.

   Copyright (C) 1992-2026 Free Software Foundation, Inc.

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

#include "event-top.h"
#include "symtab.h"
#include "gdbtypes.h"
#include "expression.h"
#include "parser-defs.h"
#include "language.h"
#include "varobj.h"
#include "m2-lang.h"
#include "c-lang.h"
#include "valprint.h"
#include "gdbarch.h"
#include "m2-exp.h"
#include "char-print.h"

/* A helper function for UNOP_HIGH.  */

struct value *
eval_op_m2_high (struct type *expect_type, struct expression *exp,
		 enum noside noside,
		 struct value *arg1)
{
  if (noside == EVAL_AVOID_SIDE_EFFECTS)
    return arg1;
  else
    {
      arg1 = coerce_ref (arg1);
      struct type *type = check_typedef (arg1->type ());

      if (m2_is_unbounded_array (type))
	{
	  struct value *temp = arg1;

	  type = type->field (1).type ();
	  /* i18n: Do not translate the "_m2_high" part!  */
	  arg1 = value_struct_elt (&temp, {}, "_m2_high", NULL,
				   _("unbounded structure "
				     "missing _m2_high field"));

	  if (arg1->type () != type)
	    arg1 = value_cast (type, arg1);
	}
    }
  return arg1;
}

/* A helper function for BINOP_SUBSCRIPT.  */

struct value *
eval_op_m2_subscript (struct type *expect_type, struct expression *exp,
		      enum noside noside,
		      struct value *arg1, struct value *arg2)
{
  /* If the user attempts to subscript something that is not an
     array or pointer type (like a plain int variable for example),
     then report this as an error.  */

  arg1 = coerce_ref (arg1);
  struct type *type = check_typedef (arg1->type ());

  if (m2_is_unbounded_array (type))
    {
      struct value *temp = arg1;
      type = type->field (0).type ();
      if (type == NULL || (type->code () != TYPE_CODE_PTR))
	error (_("internal error: unbounded "
		 "array structure is unknown"));
      /* i18n: Do not translate the "_m2_contents" part!  */
      arg1 = value_struct_elt (&temp, {}, "_m2_contents", NULL,
			       _("unbounded structure "
				 "missing _m2_contents field"));

      if (arg1->type () != type)
	arg1 = value_cast (type, arg1);

      check_typedef (arg1->type ());
      return value_ind (value_ptradd (arg1, value_as_long (arg2)));
    }
  else
    if (type->code () != TYPE_CODE_ARRAY)
      {
	if (type->name ())
	  error (_("cannot subscript something of type `%s'"),
		 type->name ());
	else
	  error (_("cannot subscript requested type"));
      }

  if (noside == EVAL_AVOID_SIDE_EFFECTS)
    return value::zero (type->target_type (), arg1->lval ());
  else
    return value_subscript (arg1, value_as_long (arg2));
}



/* Single instance of the M2 language.  */

static m2_language m2_language_defn;

/* See language.h.  */

void
m2_language::language_arch_info (struct gdbarch *gdbarch,
				 struct language_arch_info *lai) const
{
  const struct builtin_m2_type *builtin = builtin_m2_type (gdbarch);

  /* Helper function to allow shorter lines below.  */
  auto add  = [&] (struct type * t)
  {
    lai->add_primitive_type (t);
  };

  add (builtin->builtin_char);
  add (builtin->builtin_int);
  add (builtin->builtin_card);
  add (builtin->builtin_real);
  add (builtin->builtin_bool);

  lai->set_string_char_type (builtin->builtin_char);
  lai->set_bool_type (builtin->builtin_bool, "BOOLEAN");
}

class m2_wchar_printer : public wchar_printer
{
public:

  using wchar_printer::wchar_printer;

protected:

  bool printable (gdb_wchar_t w) const override
  {
    /* Historically the Modula-2 code in gdb handled \e as well.  */
    return (w == LCST ('\033')
	    || wchar_printer::printable (w));
  }

  void print_char (gdb_wchar_t w) override
  {
    if (w == LCST ('\033'))
      m_file.write (LCST ("\\e"));
    else
      wchar_printer::print_char (w);
  }
};

/* See language.h.  */

void
m2_language::printchar (int c, struct type *type,
			struct ui_file *stream) const
{
  m2_wchar_printer (type, '\'').print (c, stream);
}

/* See language.h.  */

void
m2_language::printstr (struct ui_file *stream, struct type *elttype,
			const gdb_byte *string, unsigned int length,
			const char *encoding, int force_ellipses,
			const struct value_print_options *options) const
{
  m2_wchar_printer printer (elttype, '"', encoding);
  printer.print (stream, string, length, force_ellipses, 0, options);
}

/* Called during architecture gdbarch initialisation to create language
   specific types.  */

static struct builtin_m2_type *
build_m2_types (struct gdbarch *gdbarch)
{
  struct builtin_m2_type *builtin_m2_type = new struct builtin_m2_type;

  type_allocator alloc (gdbarch);

  /* Modula-2 "pervasive" types.  NOTE:  these can be redefined!!! */
  builtin_m2_type->builtin_int
    = init_integer_type (alloc, gdbarch_int_bit (gdbarch), 0, "INTEGER");
  builtin_m2_type->builtin_card
    = init_integer_type (alloc, gdbarch_int_bit (gdbarch), 1, "CARDINAL");
  builtin_m2_type->builtin_real
    = init_float_type (alloc, gdbarch_float_bit (gdbarch), "REAL",
		       gdbarch_float_format (gdbarch));
  builtin_m2_type->builtin_char
    = init_character_type (alloc, TARGET_CHAR_BIT, 1, "CHAR");
  builtin_m2_type->builtin_bool
    = init_boolean_type (alloc, gdbarch_int_bit (gdbarch), 1, "BOOLEAN");

  return builtin_m2_type;
}

static const registry<gdbarch>::key<struct builtin_m2_type> m2_type_data;

const struct builtin_m2_type *
builtin_m2_type (struct gdbarch *gdbarch)
{
  struct builtin_m2_type *result = m2_type_data.get (gdbarch);
  if (result == nullptr)
    {
      result = build_m2_types (gdbarch);
      m2_type_data.set (gdbarch, result);
    }

  return result;
}
