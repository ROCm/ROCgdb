/* Python interface to architecture

   Copyright (C) 2013-2026 Free Software Foundation, Inc.

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

#include "gdbarch.h"
#include "arch-utils.h"
#include "disasm.h"
#include "python-internal.h"

struct arch_object : public PyObject
{
  struct gdbarch *gdbarch;

  /* Return the gdbarch associated with OBJ.  Throws an exception on
     error.  */
  struct gdbarch *require ()
  {
    if (gdbarch == nullptr)
      gdbpy_err_set_string (PyExc_RuntimeError, _("Architecture is invalid."));
    return gdbarch;
  }

  /* Implementation of gdb.Architecture.name (self) -> String.
     Returns the name of the architecture as a string value.  */
  const char *name ()
  {
    return gdbarch_bfd_arch_info (require ())->printable_name;
  }

  /* __repr__ implementation for gdb.Architecture.  */
  gdbpy_ref<> repr ();

  /* Implementation of gdb.void_type.  */
  gdbpy_ref<> void_type ()
  {
    return type_to_type_object (builtin_type (require ())->builtin_void);
  }

  /* Implementation of gdb.Architecture.register_groups (self) -> Iterator.
     Returns an iterator that will give up all valid register groups in the
     architecture SELF.  */
  gdbpy_ref<> register_groups ()
  {
    return gdbpy_new_reggroup_iterator (require ());
  }

  /* Implementation of
     gdb.Architecture.disassemble (self, start_pc [, end_pc [,count]]) -> List.
     Returns a list of instructions in a memory address range.  Each instruction
     in the list is a Python dict object.  */
  gdbpy_ref<> disassemble (gdbpy_borrowed_ref<> args,
			   gdbpy_opt_borrowed_ref<> kw);

  /* Implementation of gdb.integer_type.  */
  gdbpy_ref<> integer_type (gdbpy_borrowed_ref<> args,
			    gdbpy_opt_borrowed_ref<> kw);

  /* Implementation of gdb.Architecture.registers (self, reggroup) ->
     Iterator.  Returns an iterator over register descriptors for
     registers in GROUP within the architecture SELF.  */
  gdbpy_ref<> registers (gdbpy_borrowed_ref<> args,
			 gdbpy_opt_borrowed_ref<> kw);
};

static_assert (gdb::is_python_allocatable_v<arch_object>);

static const registry<gdbarch>::key<PyObject, gdb::noop_deleter<PyObject>>
     arch_object_data;

extern PyTypeObject arch_object_type;

/* Associates an arch_object with GDBARCH as gdbarch_data via the gdbarch
   post init registration mechanism (gdbarch_data_register_post_init).  */

static PyObject *
arch_object_data_init (struct gdbarch *gdbarch)
{
  arch_object *arch_obj = PyObject_New (arch_object, &arch_object_type);

  if (arch_obj == NULL)
    return NULL;

  arch_obj->gdbarch = gdbarch;

  return arch_obj;
}

/* Returns the struct gdbarch value corresponding to the given Python
   architecture object OBJ, which must be a gdb.Architecture object.  */

struct gdbarch *
arch_object_to_gdbarch (PyObject *obj)
{
  gdb_assert (gdbpy_is_architecture (obj));

  arch_object *py_arch = (arch_object *) obj;
  return py_arch->gdbarch;
}

/* See python-internal.h.  */

bool
gdbpy_is_architecture (PyObject *obj)
{
  return PyObject_TypeCheck (obj, &arch_object_type);
}

/* Returns the Python architecture object corresponding to GDBARCH.
   Returns a new reference to the arch_object associated as data with
   GDBARCH.  */

gdbpy_ref<>
gdbarch_to_arch_object (struct gdbarch *gdbarch)
{
  PyObject *new_ref = arch_object_data.get (gdbarch);
  if (new_ref == nullptr)
    {
      new_ref = arch_object_data_init (gdbarch);
      arch_object_data.set (gdbarch, new_ref);
    }

  /* new_ref could be NULL if creation failed.  */
  Py_XINCREF (new_ref);

  return gdbpy_ref<> (new_ref);
}

gdbpy_ref<>
arch_object::disassemble (gdbpy_borrowed_ref<> args,
			  gdbpy_opt_borrowed_ref<> kw)
{
  static const char *keywords[] = {
    "start_pc", "end_pc", "count", "styling", nullptr
  };
  CORE_ADDR pc;
  long count = 0, i;
  PyObject *start_obj = nullptr, *end_obj = nullptr, *count_obj = nullptr;
  int styling_p = 0;

  struct gdbarch *gdbarch = require ();

  gdbpy_arg_parse_tuple_and_keywords (args, kw, "O|OOp",
				      keywords, &start_obj, &end_obj,
				      &count_obj, &styling_p);

  CORE_ADDR start = gdbpy_get_address (start_obj);

  CORE_ADDR end = 0;
  if (end_obj != nullptr)
    {
      end = gdbpy_get_address (end_obj);

      if (end < start)
	gdbpy_err_set_string (PyExc_ValueError,
			      _("Argument 'end_pc' should be greater than or "
				"equal to the argument 'start_pc'."));
    }
  if (count_obj)
    {
      count = gdbpy_long_as_long (count_obj);
      if (count < 0)
	gdbpy_err_set_string (PyExc_TypeError,
			      _("Argument 'count' should be an non-negative "
				"integer."));
    }

  gdbpy_ref<> result_list = gdbpy_list_new (0);

  for (pc = start, i = 0;
       /* All args are specified.  */
       (end_obj && count_obj && pc <= end && i < count)
       /* end_pc is specified, but no count.  */
       || (end_obj && count_obj == NULL && pc <= end)
       /* end_pc is not specified, but a count is.  */
       || (end_obj == NULL && count_obj && i < count)
       /* Both end_pc and count are not specified.  */
       || (end_obj == NULL && count_obj == NULL && pc == start);)
    {
      gdbpy_ref<> insn_dict = gdbpy_dict_new ();

      gdbpy_list_append (result_list, insn_dict);

      string_file stb (styling_p);
      int insn_len = gdb_print_insn (gdbarch, pc, &stb, NULL);

      /* FIXME: Python safety.  Eventually gdb_py_object_from_ulongest
	 should throw.  */
      gdbpy_ref<> pc_obj = gdb_py_object_from_ulongest (pc);
      if (pc_obj == nullptr)
	throw gdb_python_exception ();

      gdbpy_ref<> asm_obj = gdbpy_unicode_from_string (!stb.empty ()
						       ? stb.c_str ()
						       : "<unknown>");

      /* FIXME: Python safety.  Eventually gdb_py_object_from_longest
	 should throw.  */
      gdbpy_ref<> len_obj = gdb_py_object_from_longest (insn_len);
      if (len_obj == nullptr)
	throw gdb_python_exception ();

      gdbpy_dict_set_item_string (insn_dict, "addr", pc_obj);
      gdbpy_dict_set_item_string (insn_dict, "asm", asm_obj);
      gdbpy_dict_set_item_string (insn_dict, "length", len_obj);

      pc += insn_len;
      i++;
    }

  return result_list;
}

gdbpy_ref<>
arch_object::registers (gdbpy_borrowed_ref<> args,
			gdbpy_opt_borrowed_ref<> kw)
{
  static const char *keywords[] = { "reggroup", NULL };
  const char *group_name = NULL;

  /* Parse method arguments.  */
  gdbpy_arg_parse_tuple_and_keywords (args, kw, "|s", keywords, &group_name);

  /* Extract the gdbarch from the self object.  */
  struct gdbarch *gdbarch = require ();

  return gdbpy_new_register_descriptor_iterator (gdbarch, group_name);
}

gdbpy_ref<>
arch_object::integer_type (gdbpy_borrowed_ref<> args,
			   gdbpy_opt_borrowed_ref<> kw)
{
  static const char *keywords[] = { "size", "signed", NULL };
  int size;
  PyObject *is_signed_obj = Py_True;

  gdbpy_arg_parse_tuple_and_keywords (args, kw, "i|O!", keywords, &size,
				      &PyBool_Type, &is_signed_obj);

  /* Assume signed by default.  */
  gdb_assert (PyBool_Check (is_signed_obj));
  bool is_signed = is_signed_obj == Py_True;

  struct gdbarch *gdbarch = require ();

  const struct builtin_type *builtins = builtin_type (gdbarch);
  struct type *type = nullptr;
  switch (size)
    {
    case 0:
      type = builtins->builtin_int0;
      break;
    case 8:
      type = is_signed ? builtins->builtin_int8 : builtins->builtin_uint8;
      break;
    case 16:
      type = is_signed ? builtins->builtin_int16 : builtins->builtin_uint16;
      break;
    case 24:
      type = is_signed ? builtins->builtin_int24 : builtins->builtin_uint24;
      break;
    case 32:
      type = is_signed ? builtins->builtin_int32 : builtins->builtin_uint32;
      break;
    case 64:
      type = is_signed ? builtins->builtin_int64 : builtins->builtin_uint64;
      break;
    case 128:
      type = is_signed ? builtins->builtin_int128 : builtins->builtin_uint128;
      break;

    default:
      gdbpy_err_set_string (PyExc_ValueError,
			    _("no integer type of that size is available"));
    }

  return type_to_type_object (type);
}

gdbpy_ref<>
arch_object::repr ()
{
  if (gdbarch == nullptr)
    /* FIXME: Python safety.  Eventually gdb_py_invalid_object_repr
       should throw.  */
    return gdbpy_ref<> (gdb_py_invalid_object_repr (this));

  auto arch_info = gdbarch_bfd_arch_info (gdbarch);
  return gdbpy_unicode_from_format ("<%s arch_name=%s printable_name=%s>",
				    gdbpy_py_obj_tp_name (this).c_str (),
				    arch_info->arch_name,
				    arch_info->printable_name);
}

/* Implementation of gdb.architecture_names().  Return a list of all the
   BFD architecture names that GDB understands.  */

gdbpy_ref<>
gdbpy_all_architecture_names ()
{
  gdbpy_ref<> list = gdbpy_list_new (0);

  std::vector<const char *> name_list = gdbarch_printable_names ();
  for (const char *name : name_list)
    gdbpy_list_append (list, gdbpy_unicode_from_string (name));

 return list;
}

/* Initializes the Architecture class in the gdb module.  */

static int
gdbpy_initialize_arch ()
{
  arch_object_type.tp_new = PyType_GenericNew;
  return gdbpy_type_ready (&arch_object_type);
}

GDBPY_INITIALIZE_FILE (gdbpy_initialize_arch);



static PyMethodDef arch_object_methods [] = {
  noargs_method<arch_object, &arch_object::name> ("name",
    "name () -> String.\n\
Return the name of the architecture as a string value."),
  varargs_method<arch_object, &arch_object::disassemble> ("disassemble",
    "disassemble (start_pc [, end_pc [, count]]) -> List.\n\
Return a list of at most COUNT disassembled instructions from START_PC to\n\
END_PC."),
  varargs_method<arch_object, &arch_object::integer_type> ("integer_type",
    "integer_type (size [, signed]) -> type\n\
Return an integer Type corresponding to the given bitsize and signed-ness.\n\
If not specified, the type defaults to signed."),
  noargs_method<arch_object, &arch_object::void_type> ("void_type",
    "void_type () -> type\n\
Return a void Type."),
  varargs_method<arch_object, &arch_object::registers> ("registers",
    "registers ([ group-name ]) -> Iterator.\n\
Return an iterator of register descriptors for the registers in register\n\
group GROUP-NAME."),
  noargs_method<arch_object, &arch_object::register_groups> ("register_groups",
    "register_groups () -> Iterator.\n\
Return an iterator over all of the register groups in this architecture."),
  {NULL}  /* Sentinel */
};

PyTypeObject arch_object_type = {
  PyVarObject_HEAD_INIT (NULL, 0)
  "gdb.Architecture",                 /* tp_name */
  sizeof (arch_object),               /* tp_basicsize */
  0,                                  /* tp_itemsize */
  0,                                  /* tp_dealloc */
  0,                                  /* tp_print */
  0,                                  /* tp_getattr */
  0,                                  /* tp_setattr */
  0,                                  /* tp_compare */
  wrap_tp_callback<arch_object, &arch_object::repr>, /* tp_repr */
  0,                                  /* tp_as_number */
  0,                                  /* tp_as_sequence */
  0,                                  /* tp_as_mapping */
  0,                                  /* tp_hash  */
  0,                                  /* tp_call */
  0,                                  /* tp_str */
  0,                                  /* tp_getattro */
  0,                                  /* tp_setattro */
  0,                                  /* tp_as_buffer */
  Py_TPFLAGS_DEFAULT,                 /* tp_flags */
  "GDB architecture object",          /* tp_doc */
  0,                                  /* tp_traverse */
  0,                                  /* tp_clear */
  0,                                  /* tp_richcompare */
  0,                                  /* tp_weaklistoffset */
  0,                                  /* tp_iter */
  0,                                  /* tp_iternext */
  arch_object_methods,                /* tp_methods */
  0,                                  /* tp_members */
  0,                                  /* tp_getset */
  0,                                  /* tp_base */
  0,                                  /* tp_dict */
  0,                                  /* tp_descr_get */
  0,                                  /* tp_descr_set */
  0,                                  /* tp_dictoffset */
  0,                                  /* tp_init */
  0,                                  /* tp_alloc */
};
