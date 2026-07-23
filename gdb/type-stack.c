/* Type stack for GDB parser.

   Copyright (C) 1986-2026 Free Software Foundation, Inc.

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

#include "type-stack.h"

#include "gdbtypes.h"
#include "gdbarch.h"

/* See type-stack.h.  */

void
type_stack::insert (enum type_pieces tp)
{
  gdb_assert (tp == tp_pointer || tp == tp_reference
	      || tp == tp_rvalue_reference || tp == tp_const
	      || tp == tp_volatile || tp == tp_restrict
	      || tp == tp_atomic);

  /* If there is anything on the stack (we know it will be a
     tp_pointer), insert the qualifier above it.  Otherwise, simply
     push this on the top of the stack.  */
  if (!m_elements.empty () && (tp == tp_const || tp == tp_volatile
			       || tp == tp_restrict))
    insert_into (1, tp);
  else
    insert_into (0, tp);
}

/* See type-stack.h.  */

void
type_stack::insert (struct gdbarch *gdbarch, const char *string)
{
  /* If there is anything on the stack (we know it will be a
     tp_pointer), insert the address space qualifier above it.
     Otherwise, simply push this on the top of the stack.  */
  int slot = (!m_elements.empty ()) ? 1 : 0;

  /* Check for Harvard address space delimiters and
     architecture-specific address classes.  */
  if (streq (string, "code"))
    {
      insert_into (slot, tp_harvard_aspace_identifier);
      insert_into (slot, HARVARD_ASPACE_CODE);
    }
  else if (streq (string, "data"))
    {
      insert_into (slot, tp_harvard_aspace_identifier);
      insert_into (slot, HARVARD_ASPACE_DATA);
    }
  else if (unsigned int aclass = 0;
	   gdbarch_address_class_name_to_id_p (gdbarch)
	   && gdbarch_address_class_name_to_id (gdbarch,
						string,
						aclass))
    {
      insert_into (slot, tp_aclass_identifier);
      insert_into (slot, aclass);
    }
  else
    error (_("Unknown address space/class specifier: \"%s\""), string);
}

/* See type-stack.h.  */

type_instance_flags
type_stack::follow_type_instance_flags ()
{
  type_instance_flags flags = 0;

  for (;;)
    switch (pop ())
      {
      case tp_end:
	return flags;
      case tp_const:
	flags |= TYPE_INSTANCE_FLAG_CONST;
	break;
      case tp_volatile:
	flags |= TYPE_INSTANCE_FLAG_VOLATILE;
	break;
      case tp_atomic:
	flags |= TYPE_INSTANCE_FLAG_ATOMIC;
	break;
      case tp_restrict:
	flags |= TYPE_INSTANCE_FLAG_RESTRICT;
	break;
      default:
	gdb_assert_not_reached ("unrecognized tp_ value in follow_types");
      }
}

/* See type-stack.h.  */

struct type *
type_stack::follow_types (struct type *follow_type)
{
  int done = 0;
  int make_const = 0;
  int make_volatile = 0;
  harvard_address_space make_harvard_aspace = HARVARD_ASPACE_NONE;
  int make_address_class = 0;
  bool make_restrict = false;
  bool make_atomic = false;
  int array_size;

  while (!done)
    switch (pop ())
      {
      case tp_end:
	done = 1;
	goto process_qualifiers;
	break;
      case tp_const:
	make_const = 1;
	break;
      case tp_volatile:
	make_volatile = 1;
	break;
      case tp_harvard_aspace_identifier:
	make_harvard_aspace = (harvard_address_space) pop_int ();
	break;
      case tp_aclass_identifier:
	make_address_class = pop_int ();
	break;
      case tp_atomic:
	make_atomic = true;
	break;
      case tp_restrict:
	make_restrict = true;
	break;
      case tp_pointer:
	follow_type = lookup_pointer_type (follow_type);
	goto process_qualifiers;
      case tp_reference:
	follow_type = lookup_lvalue_reference_type (follow_type);
	goto process_qualifiers;
      case tp_rvalue_reference:
	follow_type = lookup_rvalue_reference_type (follow_type);
      process_qualifiers:
	if (make_const)
	  follow_type = make_cv_type (make_const,
				      TYPE_VOLATILE (follow_type),
				      follow_type);
	if (make_volatile)
	  follow_type = make_cv_type (TYPE_CONST (follow_type),
				      make_volatile,
				      follow_type);
	if (make_harvard_aspace != HARVARD_ASPACE_NONE)
	  follow_type
	    = make_type_with_harvard_address_space (follow_type,
						    make_harvard_aspace);
	if (make_address_class != 0)
	  follow_type
	    = make_type_with_address_class (follow_type,
					    make_address_class);
	if (make_restrict)
	  follow_type = make_restrict_type (follow_type);
	if (make_atomic)
	  follow_type = make_atomic_type (follow_type);
	make_const = make_volatile = 0;
	make_harvard_aspace = HARVARD_ASPACE_NONE;
	make_address_class = 0;
	make_restrict = make_atomic = false;
	break;
      case tp_array:
	array_size = pop_int ();
	/* FIXME-type-allocation: need a way to free this type when we are
	   done with it.  */
	follow_type =
	  lookup_array_range_type (follow_type,
				   0, array_size >= 0 ? array_size - 1 : 0);
	if (array_size < 0)
	  follow_type->bounds ()->high.set_undefined ();
	break;
      case tp_function:
	/* FIXME-type-allocation: need a way to free this type when we are
	   done with it.  */
	follow_type = lookup_function_type (follow_type);
	break;

      case tp_function_with_arguments:
	{
	  std::vector<struct type *> *args = pop_typelist ();

	  follow_type
	    = lookup_function_type_with_arguments (follow_type,
						   args->size (),
						   args->data ());
	}
	break;

      case tp_type_stack:
	{
	  struct type_stack *stack = pop_type_stack ();
	  follow_type = stack->follow_types (follow_type);
	}
	break;
      default:
	gdb_assert_not_reached ("unrecognized tp_ value in follow_types");
      }
  return follow_type;
}
