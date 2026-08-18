/* Python interface to symbols.

   Copyright (C) 2008-2026 Free Software Foundation, Inc.

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

#include "block.h"
#include "frame.h"
#include "symtab.h"
#include "python-internal.h"
#include "objfiles.h"
#include "symfile.h"

struct symbol_object : public PyObject
{
  /* The GDB symbol structure this object is wrapping.  */
  struct symbol *symbol;

  /* Require a valid symbol object.  If it is not valid, throw an
     exception.  */
  void require_valid ()
  {
    if (symbol == nullptr)
      gdbpy_err_set_string (PyExc_RuntimeError, _("Symbol is invalid."));
  }

  /* Return a string representation of this symbol.  */
  const char *str ()
  {
    require_valid ();
    return symbol->print_name ();
  }

  /* 'type' attribute.  */
  gdbpy_ref<> type ();

  /* 'symtab' attribute.  */
  gdbpy_ref<> symtab ();

  /* 'name' attribute.  */
  const char *name ();

  /* 'linkage_name' attribute.  */
  const char *linkage_name ();

  /* 'addr_class' attribute.  */
  int addr_class ();

  /* 'domain' attribute.  */
  int domain ();

  /* 'is_argument' attribute.  */
  bool is_argument ();

  /* 'is_constant' attribute.  */
  bool is_constant ();

  /* 'is_function' attribute.  */
  bool is_function ();

  /* 'is_variable' attribute.  */
  bool is_variable ();

  /* 'is_artificial' attribute.  */
  bool is_artificial ();

  /* 'needs_frame' attribute.  */
  bool needs_frame ();

  /* 'line' attribute.  */
  unsigned line ();

  /* 'is_valid' method.  */
  bool is_valid ()
  {
    return symbol != nullptr;
  }

  /* 'value' method.  */
  gdbpy_ref<> value (gdbpy_borrowed_ref<> args, gdbpy_opt_borrowed_ref<> kw);

  /* "repr" implementation.  */
  gdbpy_ref<> repr ();
};

static_assert (gdb::is_python_allocatable_v<symbol_object>);

static const gdbpy_registry<gdbpy_memoizing_registry_storage<symbol_object,
  symbol, &symbol_object::symbol>> sympy_registry;

gdbpy_ref<>
symbol_object::type ()
{
  require_valid ();

  if (symbol->type () == nullptr)
    return py_none ();

  return type_to_type_object (symbol->type ());
}

gdbpy_ref<>
symbol_object::symtab ()
{
  require_valid ();

  if (!symbol->is_objfile_owned ())
    return py_none ();

  return symtab_to_symtab_object (symbol->symtab ());
}

const char *
symbol_object::name ()
{
  require_valid ();
  return symbol->natural_name ();
}

const char *
symbol_object::linkage_name ()
{
  require_valid ();
  return symbol->linkage_name ();
}

int
symbol_object::addr_class ()
{
  require_valid ();
  return symbol->loc_class ();
}

int
symbol_object::domain ()
{
  require_valid ();
  return symbol->domain ();
}

bool
symbol_object::is_argument ()
{
  require_valid ();
  return symbol->is_argument ();
}

bool
symbol_object::is_constant ()
{
  require_valid ();
  location_class loc_class = symbol->loc_class ();
  return loc_class == LOC_CONST || loc_class == LOC_CONST_BYTES;
}

bool
symbol_object::is_function ()
{
  require_valid ();
  return symbol->loc_class () == LOC_BLOCK;
}

bool
symbol_object::is_variable ()
{
  require_valid ();
  location_class loc_class = symbol->loc_class ();
  return (!symbol->is_argument ()
	  && (loc_class == LOC_LOCAL || loc_class == LOC_REGISTER
	      || loc_class == LOC_STATIC || loc_class == LOC_COMPUTED
	      || loc_class == LOC_OPTIMIZED_OUT));
}

bool
symbol_object::is_artificial ()
{
  require_valid ();
  return symbol->is_artificial ();
}

bool
symbol_object::needs_frame ()
{
  require_valid ();
  return symbol_read_needs_frame (symbol);
}

unsigned
symbol_object::line ()
{
  require_valid ();
  return symbol->line ();
}

/* Implementation of gdb.Symbol.value (self[, frame]) -> gdb.Value.  Returns
   the value of the symbol, or an error in various circumstances.  */

gdbpy_ref<>
symbol_object::value (gdbpy_borrowed_ref<> args, gdbpy_opt_borrowed_ref<> kw)
{
  frame_info_ptr frame_info = NULL;
  PyObject *frame_obj = NULL;

  static const char *keywords[] = { "frame", nullptr };
  gdbpy_arg_parse_tuple_and_keywords (args, kw, "|O!", keywords,
				      &frame_object_type, &frame_obj);

  require_valid ();
  if (symbol->loc_class () == LOC_TYPEDEF)
    gdbpy_err_set_string (PyExc_TypeError, "cannot get the value of a typedef");

  gdbpy_ref<> result;
  if (frame_obj != nullptr)
    {
      frame_info = frame_object_to_frame_info (frame_obj);
      if (frame_info == nullptr)
	error (_("invalid frame"));
    }

  if (symbol_read_needs_frame (symbol) && frame_info == nullptr)
    error (_("symbol requires a frame to compute its value"));

  /* TODO: currently, we have no way to recover the block in which SYMBOL
     was found, so we have no block to pass to read_var_value.  This will
     yield an incorrect value when symbol is not local to FRAME_INFO (this
     can happen with nested functions).  */
  scoped_value_mark free_values;
  struct value *value = read_var_value (symbol, nullptr, frame_info);
  return value_to_value_object (value);
}

/* Given a symbol, and a symbol_object that has previously been
   allocated and initialized, populate the symbol_object with the
   struct symbol data.  Also, register the symbol_object life-cycle
   with the life-cycle of the object file associated with this
   symbol, if needed.  */
static void
set_symbol (symbol_object *obj, struct symbol *symbol)
{
  obj->symbol = symbol;
  if (symbol->is_objfile_owned ())
    {
      /* Can it really happen that symbol->symtab () is NULL?  */
      if (symbol->symtab () != nullptr)
	{
	  sympy_registry.add (symbol->objfile (), obj);
	}
    }
  else
    {
      sympy_registry.add (symbol->arch (), obj);
    }
}

/* Create a new symbol object (gdb.Symbol) that encapsulates the struct
   symbol object from GDB.  */
gdbpy_ref<>
symbol_to_symbol_object (struct symbol *sym)
{
  /* Look if there's already a gdb.Symbol object for given SYMBOL
     and if so, return it.  */
  gdbpy_ref<> result;
  if (sym->is_objfile_owned ())
    result = sympy_registry.lookup (sym->objfile (), sym);
  else
    result = sympy_registry.lookup (sym->arch (), sym);
  if (result != nullptr)
    return result;

  /* FIXME: Python safety.  This should use gdbpy_new and throw on
     failure.  The callers aren't ready for this yet.  */
  symbol_object *sym_obj = PyObject_New (symbol_object, &symbol_object_type);
  if (sym_obj)
    set_symbol (sym_obj, sym);

  return gdbpy_ref<> (sym_obj);
}

/* Return the symbol that is wrapped by this symbol object.  */
struct symbol *
symbol_object_to_symbol (PyObject *obj)
{
  if (! PyObject_TypeCheck (obj, &symbol_object_type))
    return NULL;
  return ((symbol_object *) obj)->symbol;
}

static void
sympy_dealloc (PyObject *obj)
{
  symbol_object *sym_obj = (symbol_object *) obj;

  if (sym_obj->symbol != nullptr)
    {
      if (sym_obj->symbol->is_objfile_owned ())
	sympy_registry.remove (sym_obj->symbol->objfile (), sym_obj);
      else
	sympy_registry.remove (sym_obj->symbol->arch (), sym_obj);
    }

  Py_TYPE (obj)->tp_free (obj);
}

/* __repr__ implementation for gdb.Symbol.  */

gdbpy_ref<>
symbol_object::repr ()
{
  if (symbol == nullptr)
    /* FIXME: Python safety.  gdb_py_invalid_object_repr ought to
       throw on error, and return gdbpy_ref<>, but currently does
       not.  */
    return gdbpy_ref<> (gdb_py_invalid_object_repr (this));

  return gdbpy_unicode_from_format ("<%s print_name=%s>",
				    gdbpy_py_obj_tp_name (this).c_str (),
				    symbol->print_name ());
}

/* Implementation of
   gdb.lookup_symbol (name [, block] [, domain]) -> (symbol, is_field_of_this)
   A tuple with 2 elements is always returned.  The first is the symbol
   object or None, the second is a boolean with the value of
   is_a_field_of_this (see comment in lookup_symbol_in_language).  */

gdbpy_ref<>
gdbpy_lookup_symbol (gdbpy_borrowed_ref<> args, gdbpy_opt_borrowed_ref<> kw)
{
  int domain = VAR_DOMAIN;
  struct field_of_this_result is_a_field_of_this;
  const char *name;
  static const char *keywords[] = { "name", "block", "domain", NULL };
  PyObject *block_obj = nullptr;
  const struct block *block = nullptr;

  gdbpy_arg_parse_tuple_and_keywords (args, kw, "s|O!i", keywords, &name,
				      &block_object_type, &block_obj,
				      &domain);

  if (block_obj != nullptr)
    block = block_object_to_block (block_obj);
  else
    {
      frame_info_ptr selected_frame
	= get_selected_frame (_("No frame selected."));
      block = get_frame_block (selected_frame, nullptr);
    }

  domain_search_flags flags = from_scripting_domain (domain);
  struct symbol *symbol
    = lookup_symbol (name, block, flags, &is_a_field_of_this).symbol;

  gdbpy_ref<> ret_tuple = gdbpy_tuple_new (2);

  gdbpy_ref<> sym_obj;
  if (symbol)
    {
      sym_obj = symbol_to_symbol_object (symbol);
      /* FIXME: Python safety.  symbol_to_symbol_object should throw,
	 but the other callers aren't ready for this yet.  */
      if (sym_obj == nullptr)
	throw gdb_python_exception ();
    }
  else
    sym_obj = py_none ();

  gdbpy_tuple_set_item (ret_tuple, 0, std::move (sym_obj));

  gdbpy_ref<> bool_obj
    = gdbpy_bool_from_long (is_a_field_of_this.type != nullptr);
  gdbpy_tuple_set_item (ret_tuple, 1, std::move (bool_obj));

  return ret_tuple;
}

/* Implementation of
   gdb.lookup_global_symbol (name [, domain]) -> symbol or None.  */

gdbpy_ref<>
gdbpy_lookup_global_symbol (gdbpy_borrowed_ref<> args,
			    gdbpy_opt_borrowed_ref<> kw)
{
  int domain = VAR_DOMAIN;
  const char *name;
  static const char *keywords[] = { "name", "domain", NULL };

  gdbpy_arg_parse_tuple_and_keywords (args, kw, "s|i", keywords, &name,
				      &domain);

  domain_search_flags flags = from_scripting_domain (domain);
  struct symbol *symbol = lookup_global_symbol (name, NULL, flags).symbol;

  gdbpy_ref<> sym_obj;
  if (symbol != nullptr)
    {
      /* FIXME: Python safety.  symbol_to_symbol_object should throw,
	 but the other callers aren't ready for this yet.  */
      sym_obj = symbol_to_symbol_object (symbol);
      if (sym_obj == nullptr)
	throw gdb_python_exception ();
    }
  else
    sym_obj = py_none ();

  return sym_obj;
}

/* Implementation of
   gdb.lookup_static_symbol (name [, domain]) -> symbol or None.  */

gdbpy_ref<>
gdbpy_lookup_static_symbol (gdbpy_borrowed_ref<> args,
			    gdbpy_opt_borrowed_ref<> kw)
{
  const char *name;
  int domain = VAR_DOMAIN;
  static const char *keywords[] = { "name", "domain", NULL };
  struct symbol *symbol = NULL;

  gdbpy_arg_parse_tuple_and_keywords (args, kw, "s|i", keywords, &name,
				      &domain);

  /* In order to find static symbols associated with the "current" object
     file ahead of those from other object files, we first need to see if
     we can acquire a current block.  If this fails however, then we still
     want to search all static symbols, so don't throw an exception just
     yet.  */
  const struct block *block = NULL;
  try
    {
      frame_info_ptr selected_frame
	= get_selected_frame (_("No frame selected."));
      block = get_frame_block (selected_frame, NULL);
    }
  catch (const gdb_exception_error &except)
    {
      /* Nothing.  */
    }

  domain_search_flags flags = from_scripting_domain (domain);

  if (block != nullptr)
    symbol = lookup_symbol_in_static_block (name, block, flags).symbol;

  if (symbol == nullptr)
    symbol = lookup_static_symbol (name, flags).symbol;

  gdbpy_ref<> sym_obj;
  if (symbol != nullptr)
    {
      /* FIXME: Python safety.  symbol_to_symbol_object should throw,
	 but the other callers aren't ready for this yet.  */
      sym_obj = symbol_to_symbol_object (symbol);
      if (sym_obj == nullptr)
	throw gdb_python_exception ();
    }
  else
    sym_obj = py_none ();

  return sym_obj;
}

/* Implementation of
   gdb.lookup_static_symbols (name [, domain]) -> symbol list.

   Returns a list of all static symbols matching NAME in DOMAIN.  */

gdbpy_ref<>
gdbpy_lookup_static_symbols (gdbpy_borrowed_ref<> args,
			     gdbpy_opt_borrowed_ref<> kw)
{
  const char *name;
  int domain = VAR_DOMAIN;
  static const char *keywords[] = { "name", "domain", NULL };

  gdbpy_arg_parse_tuple_and_keywords (args, kw, "s|i", keywords, &name,
				      &domain);

  gdbpy_ref<> return_list = gdbpy_new_list (0);

  domain_search_flags flags = from_scripting_domain (domain);

  /* Expand any symtabs that contain potentially matching symbols.  */
  lookup_name_info lookup_name (name, symbol_name_match_type::FULL);

  for (objfile &objfile : current_program_space->objfiles ())
    {
      auto callback = [&] (compunit_symtab *cust)
      {
	/* Skip included compunits to prevent including compunits from
	   being searched twice.  */
	if (cust->user != nullptr)
	  return iteration_status::keep_going;

	const struct blockvector *bv = cust->blockvector ();
	const struct block *block = bv->static_block ();

	if (block != nullptr)
	  {
	    symbol *symbol = lookup_symbol_in_static_block
	      (name, block, flags).symbol;

	    if (symbol != nullptr)
	      {
		/* FIXME: Python safety.  symbol_to_symbol_object should throw,
		   but the other callers aren't ready for this yet.  */
		gdbpy_ref<> sym_obj = symbol_to_symbol_object (symbol);
		if (sym_obj == nullptr)
		  throw gdb_python_exception ();
		gdbpy_list_append (return_list, sym_obj);
	      }
	  }

	return iteration_status::keep_going;
      };

      /* The callback will throw on any error, so iteration should
	 never stop unexpectedly.  */
      iteration_status status = objfile.search (nullptr, &lookup_name,
						nullptr, callback,
						SEARCH_STATIC_BLOCK, flags);
      gdb_assert (status == iteration_status::keep_going);
    }

  return return_list;
}

static int
gdbpy_initialize_symbols ()
{
  if (gdbpy_type_ready (&symbol_object_type) < 0)
    return -1;

  if (PyModule_AddIntConstant (gdb_module, "SYMBOL_LOC_UNDEF", LOC_UNDEF) < 0
      || PyModule_AddIntConstant (gdb_module, "SYMBOL_LOC_CONST",
				  LOC_CONST) < 0
      || PyModule_AddIntConstant (gdb_module, "SYMBOL_LOC_STATIC",
				  LOC_STATIC) < 0
      || PyModule_AddIntConstant (gdb_module, "SYMBOL_LOC_REGISTER",
				  LOC_REGISTER) < 0
      || PyModule_AddIntConstant (gdb_module, "SYMBOL_LOC_ARG",
				  LOC_ARG) < 0
      || PyModule_AddIntConstant (gdb_module, "SYMBOL_LOC_REF_ARG",
				  LOC_REF_ARG) < 0
      || PyModule_AddIntConstant (gdb_module, "SYMBOL_LOC_LOCAL",
				  LOC_LOCAL) < 0
      || PyModule_AddIntConstant (gdb_module, "SYMBOL_LOC_TYPEDEF",
				  LOC_TYPEDEF) < 0
      || PyModule_AddIntConstant (gdb_module, "SYMBOL_LOC_LABEL",
				  LOC_LABEL) < 0
      || PyModule_AddIntConstant (gdb_module, "SYMBOL_LOC_BLOCK",
				  LOC_BLOCK) < 0
      || PyModule_AddIntConstant (gdb_module, "SYMBOL_LOC_CONST_BYTES",
				  LOC_CONST_BYTES) < 0
      || PyModule_AddIntConstant (gdb_module, "SYMBOL_LOC_UNRESOLVED",
				  LOC_UNRESOLVED) < 0
      || PyModule_AddIntConstant (gdb_module, "SYMBOL_LOC_OPTIMIZED_OUT",
				  LOC_OPTIMIZED_OUT) < 0
      || PyModule_AddIntConstant (gdb_module, "SYMBOL_LOC_COMPUTED",
				  LOC_COMPUTED) < 0
      || PyModule_AddIntConstant (gdb_module, "SYMBOL_LOC_COMMON_BLOCK",
				  LOC_COMMON_BLOCK) < 0
      || PyModule_AddIntConstant (gdb_module, "SYMBOL_LOC_REGPARM_ADDR",
				  LOC_REGPARM_ADDR) < 0)
    return -1;

#define SYM_DOMAIN(X)							\
  if (PyModule_AddIntConstant (gdb_module, "SYMBOL_" #X "_DOMAIN",	\
			       to_scripting_domain (X ## _DOMAIN)) < 0	\
      || PyModule_AddIntConstant (gdb_module, "SEARCH_" #X "_DOMAIN",	\
				  to_scripting_domain (SEARCH_ ## X ## _DOMAIN)) < 0) \
    return -1;
#include "sym-domains.def"
#undef SYM_DOMAIN

  return 0;
}

GDBPY_INITIALIZE_FILE (gdbpy_initialize_symbols);



static gdb_PyGetSetDef symbol_object_getset[] = {
  { "type", wrap_getter<symbol_object, &symbol_object::type>, NULL,
    "Type of the symbol.", NULL },
  { "symtab", wrap_getter<symbol_object, &symbol_object::symtab>, NULL,
    "Symbol table in which the symbol appears.", NULL },
  { "name", wrap_getter<symbol_object, &symbol_object::name>, NULL,
    "Name of the symbol, as it appears in the source code.", NULL },
  { "linkage_name", wrap_getter<symbol_object, &symbol_object::linkage_name>,
    NULL, "Name of the symbol, as used by the linker (i.e., may be mangled).",
    NULL },
  { "print_name", wrap_getter<symbol_object, &symbol_object::str>, NULL,
    "Name of the symbol in a form suitable for output.\n\
This is either name or linkage_name, depending on whether the user asked GDB\n\
to display demangled or mangled names.", NULL },
  { "addr_class", wrap_getter<symbol_object, &symbol_object::addr_class>, NULL,
    "Address class of the symbol." },
  { "domain", wrap_getter<symbol_object, &symbol_object::domain>, nullptr,
    "Domain of the symbol." },
  { "is_argument", wrap_getter<symbol_object, &symbol_object::is_argument>,
    NULL, "True if the symbol is an argument of a function." },
  { "is_artificial", wrap_getter<symbol_object, &symbol_object::is_artificial>,
    nullptr, "True if the symbol is marked artificial." },
  { "is_constant", wrap_getter<symbol_object, &symbol_object::is_constant>,
    NULL, "True if the symbol is a constant." },
  { "is_function", wrap_getter<symbol_object, &symbol_object::is_function>,
    NULL, "True if the symbol is a function or method." },
  { "is_variable", wrap_getter<symbol_object, &symbol_object::is_variable>,
    NULL, "True if the symbol is a variable." },
  { "needs_frame", wrap_getter<symbol_object, &symbol_object::needs_frame>,
    NULL, "True if the symbol requires a frame for evaluation." },
  { "line", wrap_getter<symbol_object, &symbol_object::line>, NULL,
    "The source line number at which the symbol was defined." },
  { NULL }  /* Sentinel */
};

static PyMethodDef symbol_object_methods[] = {
  noargs_method<symbol_object, &symbol_object::is_valid> ("is_valid",
    "is_valid () -> Boolean.\n\
Return true if this symbol is valid, false if not."),
  varargs_method<symbol_object, &symbol_object::value> ("value",
    "value ([frame]) -> gdb.Value\n\
Return the value of the symbol."),
  {NULL}  /* Sentinel */
};

PyTypeObject symbol_object_type = {
  PyVarObject_HEAD_INIT (NULL, 0)
  "gdb.Symbol",			  /*tp_name*/
  sizeof (symbol_object),	  /*tp_basicsize*/
  0,				  /*tp_itemsize*/
  sympy_dealloc,		  /*tp_dealloc*/
  0,				  /*tp_print*/
  0,				  /*tp_getattr*/
  0,				  /*tp_setattr*/
  0,				  /*tp_compare*/
  wrap_tp_callback<symbol_object, &symbol_object::repr>, /*tp_repr*/
  0,				  /*tp_as_number*/
  0,				  /*tp_as_sequence*/
  0,				  /*tp_as_mapping*/
  0,				  /*tp_hash */
  0,				  /*tp_call*/
  wrap_tp_callback<symbol_object, &symbol_object::str>, /*tp_str*/
  0,				  /*tp_getattro*/
  0,				  /*tp_setattro*/
  0,				  /*tp_as_buffer*/
  Py_TPFLAGS_DEFAULT,		  /*tp_flags*/
  "GDB symbol object",		  /*tp_doc */
  0,				  /*tp_traverse */
  0,				  /*tp_clear */
  0,				  /*tp_richcompare */
  0,				  /*tp_weaklistoffset */
  0,				  /*tp_iter */
  0,				  /*tp_iternext */
  symbol_object_methods,	  /*tp_methods */
  0,				  /*tp_members */
  symbol_object_getset		  /*tp_getset */
};
