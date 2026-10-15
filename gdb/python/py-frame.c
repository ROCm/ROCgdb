/* Python interface to stack frames

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

#include "language.h"
#include "charset.h"
#include "block.h"
#include "frame.h"
#include "symtab.h"
#include "stack.h"
#include "value.h"
#include "python-internal.h"
#include "symfile.h"
#include "objfiles.h"

struct frame_object : public PyObject
{
  struct frame_id frame_id;
  struct gdbarch *gdbarch;

  /* Marks that the FRAME_ID member actually holds the ID of the frame next
     to this, and not this frames' ID itself.  This is a hack to permit Python
     frame objects which represent invalid frames (i.e., the last frame_info
     in a corrupt stack).  The problem arises from the fact that this code
     relies on FRAME_ID to uniquely identify a frame, which is not always true
     for the last "frame" in a corrupt stack (it can have a null ID, or the same
     ID as the  previous frame).  Whenever get_prev_frame returns NULL, we
     record the frame_id of the next frame and set FRAME_ID_IS_NEXT to 1.  */
  int frame_id_is_next;

  /* Return the associated frame.  Throws an exception on error.  */
  frame_info_ptr require_frame ()
  {
    frame_info_ptr frame = frame_object_to_frame_info (this);
    /* FIXME: Python safety.  frame_object_to_frame_info should be
       converted, but the callers aren't ready yet.  */
    if (frame == nullptr)
      gdbpy_err_format (PyExc_RuntimeError, _("Frame is invalid."));
    return frame;
  }

  /* Called by the Python interpreter to obtain string representation
     of the object.  */
  gdbpy_ref<> str ()
  {
    return gdbpy_unicode_from_string (frame_id.to_string ());
  }

  /* Implement repr() for gdb.Frame.  */
  gdbpy_ref<> repr ();

  /* Implementation of gdb.Frame.is_valid (self) -> Boolean.  Returns
     True if the frame corresponding to the frame_id of this object
     still exists in the inferior.  */
  bool is_valid ()
  {
    return frame_object_to_frame_info (this) != nullptr;
  }

  /* Implementation of gdb.Frame.name (self) -> String.
     Returns the name of the function corresponding to this frame.  */
  gdb::unique_xmalloc_ptr<char> name ();

  /* Implementation of gdb.Frame.type (self) -> Integer.
     Returns the frame type, namely one of the gdb.*_FRAME constants.  */
  ULONGEST type ();

  /* Implementation of gdb.Frame.architecture (self) -> gdb.Architecture.
     Returns the frame's architecture as a gdb.Architecture object.  */
  gdbpy_ref<> arch ();

  /* Implementation of gdb.Frame.unwind_stop_reason (self) -> Integer.
     Returns one of the gdb.FRAME_UNWIND_* constants.  */
  int unwind_stop_reason ();

  /* Implementation of gdb.Frame.pc (self) -> Long.
     Returns the frame's resume address.  */
  ULONGEST pc ();

  /* Implementation of gdb.Frame.read_register (self, register) -> gdb.Value.
     Returns the value of a register in this frame.  */
  gdbpy_ref<> read_register (gdbpy_borrowed_ref<> args,
			     gdbpy_opt_borrowed_ref<> kw);

  /* Implementation of gdb.Frame.block (self) -> gdb.Block.
     Returns the frame's code block.  */
  gdbpy_ref<> block ();

  /* Implementation of gdb.Frame.function (self) -> gdb.Symbol.
     Returns the symbol for the function corresponding to this frame.  */
  gdbpy_ref<> function ();

  /* Implementation of gdb.Frame.older (self) -> gdb.Frame.
     Returns the frame immediately older (outer) to this frame, or None if
     there isn't one.  */
  gdbpy_ref<> older ();

  /* Implementation of gdb.Frame.newer (self) -> gdb.Frame.
     Returns the frame immediately newer (inner) to this frame, or None if
     there isn't one.  */
  gdbpy_ref<> newer ();

  /* Implementation of gdb.Frame.find_sal (self) -> gdb.Symtab_and_line.
     Returns the frame's symtab and line.  */
  gdbpy_ref<> find_sal ();

  /* Implementation of gdb.Frame.read_var_value (self, variable,
     [block]) -> gdb.Value.  If the optional block argument is provided
     start the search from that block, otherwise search from the frame's
     current block (determined by examining the resume address of the
     frame).  The variable argument must be a string or an instance of a
     gdb.Symbol.  The block argument must be an instance of gdb.Block.  Returns
     NULL on error, with a python exception set.  */
  gdbpy_ref<> read_var (gdbpy_borrowed_ref<> args,
			gdbpy_opt_borrowed_ref<> kw);

  /* Select this frame.  */
  void select ();

  /* The stack frame level for this frame.  */
  int level ();

  /* The language for this frame.  */
  const char *language ();

  /* The static link for this frame.  */
  gdbpy_ref<> static_link ();

  /* Implementation of the Python richcompare API.  */
  std::optional<bool> richcompare (gdbpy_borrowed_ref<> other, int op);

  static PyTypeObject *corresponding_object_type;
};

static_assert (gdb::is_python_allocatable_v<frame_object>);

/* Returns the frame_info object corresponding to the given Python Frame
   object.  If the frame doesn't exist anymore (the frame id doesn't
   correspond to any frame in the inferior), returns NULL.  */
/* FIXME: Python safety.  This function should be converted, but the
   callers aren't ready yet.  */

frame_info_ptr
frame_object_to_frame_info (PyObject *obj)
{
  frame_object *frame_obj = (frame_object *) obj;
  frame_info_ptr frame;

  frame = frame_find_by_id (frame_obj->frame_id);
  if (frame == NULL)
    return NULL;

  return frame;
}

gdbpy_ref<>
frame_object::repr ()
{
  frame_info_ptr f_info = frame_find_by_id (frame_id);
  if (f_info == nullptr)
    /* FIXME: Python safety.  gdb_py_invalid_object_repr should
       throw on error. */
    return gdbpy_ref<> (gdb_py_invalid_object_repr (this));

  return gdbpy_unicode_from_format ("<%s level=%d frame-id=%s>",
				    gdbpy_py_obj_tp_name (this).c_str (),
				    frame_relative_level (f_info),
				    frame_id.to_string ().c_str ());
}

gdb::unique_xmalloc_ptr<char>
frame_object::name ()
{
  frame_info_ptr frame = require_frame ();
  enum language lang;
  return find_frame_funname (frame, &lang, nullptr);
}

ULONGEST
frame_object::type ()
{
  frame_info_ptr frame = require_frame ();
  return get_frame_type (frame);
}

gdbpy_ref<>
frame_object::arch ()
{
  require_frame ();
  return gdbarch_to_arch_object (gdbarch);
}

int
frame_object::unwind_stop_reason ()
{
  frame_info_ptr frame = require_frame ();
  return get_frame_unwind_stop_reason (frame);
}

ULONGEST
frame_object::pc ()
{
  frame_info_ptr frame = require_frame ();
  return get_frame_pc (frame);
}

gdbpy_ref<>
frame_object::read_register (gdbpy_borrowed_ref<> args,
			     gdbpy_opt_borrowed_ref<> kw)
{
  PyObject *pyo_reg_id;

  static const char *keywords[] = { "register", nullptr };
  gdbpy_arg_parse_tuple_and_keywords (args, kw, "O", keywords, &pyo_reg_id);

  scoped_value_mark free_values;
  frame_info_ptr frame = require_frame ();

  int regnum;
  if (!gdbpy_parse_register_id (get_frame_arch (frame), pyo_reg_id, &regnum))
    {
      /* FIXME: Python safety.  gdbpy_parse_register_id should throw
	 on error.  */
      throw gdb_python_exception ();
    }

  gdb_assert (regnum >= 0);
  value *val
    = value_of_register (regnum, get_next_frame_sentinel_okay (frame));

  if (val == nullptr)
    gdbpy_err_set_string (PyExc_ValueError, _("Can't read register."));

  return value_to_value_object (val);
}

gdbpy_ref<>
frame_object::block ()
{
  frame_info_ptr frame = require_frame ();
  const struct block *block = get_frame_block (frame, nullptr);

  const struct block *fn_block;
  for (fn_block = block;
       fn_block != NULL && fn_block->function () == NULL;
       fn_block = fn_block->superblock ())
    ;

  if (block == NULL || fn_block == NULL || fn_block->function () == NULL)
    gdbpy_err_set_string (PyExc_RuntimeError,
			  _("Cannot locate block for frame."));

  return block_to_block_object (block, fn_block->function ()->objfile ());
}


gdbpy_ref<>
frame_object::function ()
{
  frame_info_ptr frame = require_frame ();

  struct symbol *sym = nullptr;
  enum language funlang;
  gdb::unique_xmalloc_ptr<char> funname
    = find_frame_funname (frame, &funlang, &sym);

  if (sym != nullptr)
    return symbol_to_symbol_object (sym);

  return py_none ();
}

/* Convert a frame_info struct to a Python Frame object.
   Sets a Python exception and returns NULL on error.  */
/* FIXME: Python safety.  This function should be converted, but the
   callers aren't ready yet.  */

gdbpy_ref<>
frame_info_to_frame_object (const frame_info_ptr &frame)
{
  gdbpy_ref<frame_object> frame_obj (PyObject_New (frame_object,
						   &frame_object_type));
  if (frame_obj == NULL)
    return NULL;

  try
    {
      frame_obj->frame_id = get_frame_id (frame);
      frame_obj->gdbarch = get_frame_arch (frame);
    }
  catch (const gdb_exception &except)
    {
      return gdbpy_handle_gdb_exception (nullptr, except);
    }

  return frame_obj;
}

gdbpy_ref<>
frame_object::older ()
{
  frame_info_ptr frame = require_frame ();
  frame_info_ptr prev = get_prev_frame (frame);

  if (prev)
    return frame_info_to_frame_object (prev);

  return py_none ();
}

gdbpy_ref<>
frame_object::newer ()
{
  frame_info_ptr frame = require_frame ();
  frame_info_ptr next = get_next_frame (frame);

  if (next)
    return frame_info_to_frame_object (next);

  return py_none ();
}

gdbpy_ref<>
frame_object::find_sal ()
{
  frame_info_ptr frame = require_frame ();
  symtab_and_line sal = find_frame_sal (frame);
  return symtab_and_line_to_sal_object (sal);
}

gdbpy_ref<>
frame_object::read_var (gdbpy_borrowed_ref<> args,
			gdbpy_opt_borrowed_ref<> kw)
{
  PyObject *sym_obj, *block_obj = NULL;

  static const char *keywords[] = { "variable", "block", nullptr };
  gdbpy_arg_parse_tuple_and_keywords (args, kw, "O|O!", keywords,
				      &sym_obj, &block_object_type,
				      &block_obj);

  const struct block *block = NULL;
  struct symbol *var = NULL;	/* gcc-4.3.2 false warning.  */
  if (PyObject_TypeCheck (sym_obj, &symbol_object_type))
    var = symbol_object_to_symbol (sym_obj);
  else if (gdbpy_is_string (sym_obj))
    {
      gdb::unique_xmalloc_ptr<char>
	var_name (python_string_to_target_string (sym_obj));

      /* FIXME: Python safety.  python_string_to_target_string should
	 throw on error.  */
      if (var_name == nullptr)
	throw gdb_python_exception ();

      if (block_obj != nullptr)
	{
	  /* This call should only fail if the type of BLOCK_OBJ is wrong,
	     and we ensure the type is correct when we parse the arguments,
	     so we can just assert the return value is not nullptr.  */
	  block = block_object_to_block (block_obj);
	  gdb_assert (block != nullptr);
	}

      frame_info_ptr frame = require_frame ();

      if (!block)
	block = get_frame_block (frame, NULL);
      block_symbol lookup_sym = lookup_symbol (var_name.get (), block,
					       SEARCH_VFT, nullptr);
      var = lookup_sym.symbol;
      block = lookup_sym.block;

      if (var == nullptr)
	gdbpy_err_format (PyExc_ValueError,
			  _("Variable '%s' not found."), var_name.get ());
    }
  else
    gdbpy_err_format (PyExc_TypeError,
		      _("argument 1 must be gdb.Symbol or str, not %s"),
		      gdbpy_py_obj_tp_name (sym_obj).c_str ());

  frame_info_ptr frame = require_frame ();
  scoped_value_mark free_values;
  struct value *val = read_var_value (var, block, frame);
  return value_to_value_object (val);
}

void
frame_object::select ()
{
  frame_info_ptr fi = require_frame ();
  select_frame (fi);
}

int
frame_object::level ()
{
  frame_info_ptr fi = require_frame ();
  return frame_relative_level (fi);
}

const char *
frame_object::language ()
{
  frame_info_ptr fi = require_frame ();

  enum language lang = get_frame_language (fi);
  const language_defn *lang_def = language_def (lang);

  return lang_def->name ();
}

gdbpy_ref<>
frame_object::static_link ()
{
  frame_info_ptr link = require_frame ();
  link = frame_follow_static_link (link);

  if (link == nullptr)
    return py_none ();

  return frame_info_to_frame_object (link);
}

/* Implementation of gdb.newest_frame () -> gdb.Frame.
   Returns the newest frame object.  */

gdbpy_ref<>
gdbpy_newest_frame ()
{
  /* FIXME: Python safety.  Convert frame_info_to_frame_object.  */
  gdbpy_ref<> result = frame_info_to_frame_object (get_current_frame ());
  if (result == nullptr)
    throw gdb_python_exception ();
  return result;
}

/* Implementation of gdb.selected_frame () -> gdb.Frame.
   Returns the selected frame object.  */

gdbpy_ref<>
gdbpy_selected_frame ()
{
  frame_info_ptr frame
    = get_selected_frame ("No frame is currently selected.");
  /* FIXME: Python safety.  Convert frame_info_to_frame_object.  */
  gdbpy_ref<> result = frame_info_to_frame_object (frame);
  if (result == nullptr)
    throw gdb_python_exception ();
  return result;
}

/* Implementation of gdb.stop_reason_string (Integer) -> String.
   Return a string explaining the unwind stop reason.  */

const char *
gdbpy_frame_stop_reason_string (gdbpy_borrowed_ref<> args,
				gdbpy_opt_borrowed_ref<> kw)
{
  int reason;

  static const char *keywords[] = { "reason", nullptr };
  gdbpy_arg_parse_tuple_and_keywords (args, kw, "i", keywords, &reason);

  if (reason < UNWIND_FIRST || reason > UNWIND_LAST)
    gdbpy_err_set_string (PyExc_ValueError,
			  _("Invalid frame stop reason."));

  return unwind_stop_reason_to_string ((enum unwind_stop_reason) reason);
}

/* Implements the equality comparison for Frame objects.  */

std::optional<bool>
frame_object::richcompare (gdbpy_borrowed_ref<> other, int op)
{
  int result;

  if (!PyObject_TypeCheck (other, &frame_object_type)
      || (op != Py_EQ && op != Py_NE))
    return std::nullopt;

  frame_object *other_frame = other;

  if (frame_id == other_frame->frame_id)
    result = Py_EQ;
  else
    result = Py_NE;

  return op == result;
}

PyTypeObject *frame_object::corresponding_object_type = &frame_object_type;

/* Sets up the Frame API in the gdb module.  */

static int
gdbpy_initialize_frames ()
{
  frame_object_type.tp_new = PyType_GenericNew;
  if (gdbpy_type_ready (&frame_object_type) < 0)
    return -1;

  /* Note: These would probably be best exposed as class attributes of
     Frame, but I don't know how to do it except by messing with the
     type's dictionary.  That seems too messy.  */
  if (PyModule_AddIntConstant (gdb_module, "NORMAL_FRAME", NORMAL_FRAME) < 0
      || PyModule_AddIntConstant (gdb_module, "DUMMY_FRAME", DUMMY_FRAME) < 0
      || PyModule_AddIntConstant (gdb_module, "INLINE_FRAME", INLINE_FRAME) < 0
      || PyModule_AddIntConstant (gdb_module, "TAILCALL_FRAME",
				  TAILCALL_FRAME) < 0
      || PyModule_AddIntConstant (gdb_module, "SIGTRAMP_FRAME",
				  SIGTRAMP_FRAME) < 0
      || PyModule_AddIntConstant (gdb_module, "ARCH_FRAME", ARCH_FRAME) < 0
      || PyModule_AddIntConstant (gdb_module, "SENTINEL_FRAME",
				  SENTINEL_FRAME) < 0)
    return -1;

#define SET(name, description) \
  if (PyModule_AddIntConstant (gdb_module, "FRAME_"#name, name) < 0) \
    return -1;
#include "unwind_stop_reasons.def"
#undef SET

  return 0;
}

GDBPY_INITIALIZE_FILE (gdbpy_initialize_frames);



static PyMethodDef frame_object_methods[] = {
  noargs_method<frame_object, &frame_object::is_valid> ("is_valid",
    "is_valid () -> Boolean.\n\
Return true if this frame is valid, false if not."),
  noargs_method<frame_object, &frame_object::name> ("name",
    "name () -> String.\n\
Return the function name of the frame, or None if it can't be determined."),
  noargs_method<frame_object, &frame_object::type> ("type",
    "type () -> Integer.\n\
Return the type of the frame."),
  noargs_method<frame_object, &frame_object::arch> ("architecture",
    "architecture () -> gdb.Architecture.\n\
Return the architecture of the frame."),
  noargs_method<frame_object, &frame_object::unwind_stop_reason>
    ("unwind_stop_reason",
     "unwind_stop_reason () -> Integer.\n\
Return the reason why it's not possible to find frames older than this."),
  noargs_method<frame_object, &frame_object::pc> ("pc",
    "pc () -> Long.\n\
Return the frame's resume address."),
  varargs_method<frame_object, &frame_object::read_register> ("read_register",
    "read_register (register_name) -> gdb.Value\n\
Return the value of the register in the frame."),
  noargs_method<frame_object, &frame_object::block> ("block",
    "block () -> gdb.Block.\n\
Return the frame's code block."),
  noargs_method<frame_object, &frame_object::function> ("function",
    "function () -> gdb.Symbol.\n\
Returns the symbol for the function corresponding to this frame."),
  noargs_method<frame_object, &frame_object::older> ("older",
    "older () -> gdb.Frame.\n\
Return the frame that called this frame."),
  noargs_method<frame_object, &frame_object::newer> ("newer",
    "newer () -> gdb.Frame.\n\
Return the frame called by this frame."),
  noargs_method<frame_object, &frame_object::find_sal> ("find_sal",
    "find_sal () -> gdb.Symtab_and_line.\n\
Return the frame's symtab and line."),
  varargs_method<frame_object, &frame_object::read_var> ("read_var",
    "read_var (variable) -> gdb.Value.\n\
Return the value of the variable in this frame."),
  noargs_method<frame_object, &frame_object::select> ("select",
    "Select this frame as the user's current frame."),
  noargs_method<frame_object, &frame_object::level> ("level",
    "The stack level of this frame."),
  noargs_method<frame_object, &frame_object::language> ("language",
    "The language of this frame."),
  noargs_method<frame_object, &frame_object::static_link> ("static_link",
    "The static link of this frame, or None."),
  {NULL}  /* Sentinel */
};

PyTypeObject frame_object_type = {
  PyVarObject_HEAD_INIT (NULL, 0)
  "gdb.Frame",			  /* tp_name */
  sizeof (frame_object),	  /* tp_basicsize */
  0,				  /* tp_itemsize */
  0,				  /* tp_dealloc */
  0,				  /* tp_print */
  0,				  /* tp_getattr */
  0,				  /* tp_setattr */
  0,				  /* tp_compare */
  wrap_tp_callback<frame_object, &frame_object::repr>, /* tp_repr */
  0,				  /* tp_as_number */
  0,				  /* tp_as_sequence */
  0,				  /* tp_as_mapping */
  0,				  /* tp_hash  */
  0,				  /* tp_call */
  wrap_tp_callback<frame_object, &frame_object::str>, /* tp_str */
  0,				  /* tp_getattro */
  0,				  /* tp_setattro */
  0,				  /* tp_as_buffer */
  Py_TPFLAGS_DEFAULT,		  /* tp_flags */
  "GDB frame object",		  /* tp_doc */
  0,				  /* tp_traverse */
  0,				  /* tp_clear */
  wrap_richcompare<frame_object, &frame_object::richcompare>, /* tp_richcompare */
  0,				  /* tp_weaklistoffset */
  0,				  /* tp_iter */
  0,				  /* tp_iternext */
  frame_object_methods,		  /* tp_methods */
  0,				  /* tp_members */
  0,				  /* tp_getset */
  0,				  /* tp_base */
  0,				  /* tp_dict */
  0,				  /* tp_descr_get */
  0,				  /* tp_descr_set */
  0,				  /* tp_dictoffset */
  0,				  /* tp_init */
  0,				  /* tp_alloc */
};
