/* Wrappers for some Python safety.

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

#ifndef GDB_PYTHON_PY_SAFETY_H
#define GDB_PYTHON_PY_SAFETY_H

#include "gdbsupport/traits.h"
#include "py-ref.h"
#include "py-wrappers.h"
#include "charset.h"

/* This file holds wrapper templates for the various ways that gdb
   code might be exposed to Python.  These wrappers are part of gdb's
   "Python safety" approach -- utilities designed to try to prevent
   refcount problems, missing error checks, and that also remove the
   need to wrap calls into gdb in an explicit try/catch.

   See py-wrappers.h for some more discussion of this.

   Implementation functions and methods -- the stuff you write to
   expose some part of gdb to Python -- are written in a certain
   style.

   An implementation function is a module-level function.  For example
   something like 'gdb.register_window_type' is implemented as a
   function.  An implementation method is a method of the gdb class
   implementing a certain Python type; for example
   'gdb.TuiWindow.width' is implemented via a method of
   gdbpy_tui_window.

   Each of these will accept gdbpy_borrowed_ref arguments (or in some
   more limited situations, a gdbpy_opt_borrowed_ref) and return any
   relevant type, which will be automatically converted (see the
   'to_python' overloads below) to the correct Python type.  Note that
   if the 'to_python' functions aren't appropriate in your case, you
   can always use the escape hatch of returning a gdbpy_ref<> and
   handling type conversion manually.

   There are some constexpr functions below that wrap the
   implementation methods, and create PyMethodDef objects and the
   like.

   Implementation methods are expected to use the wrappers in
   py-wrappers.h and not generally call into Python directly.

   Implementation details of the method-wrapping safety code are put
   into this namespace, just to emphasize that these shouldn't be used
   elsewhere.  Skip past the namespace to find the public APIs.  */
namespace safety_details
{
/* Overloads of "to_python" are used by the safety wrappers to convert
   a function's real return value to a Python object.  A new reference
   will always be returned.  These are a detail of the method-wrapping
   code.  Note that unlike the "safe" wrapper APIs that are commonly
   used in gdb, these are all Python-facing and will return NULL on
   error.  */

static inline PyObject *
to_python (bool value)
{
  return PyBool_FromLong (value);
}

template<typename T, typename = gdb::Requires<std::is_integral<T>>>
static inline PyObject *
to_python (T value)
{
  if constexpr (std::is_signed<T>::value)
    return gdb_py_object_from_longest (value).release ();
  else
    return gdb_py_object_from_ulongest (value).release ();
}

static inline PyObject *
to_python (const char *value)
{
  if (value == nullptr)
    return py_none ().release ();
  return PyUnicode_Decode (value, strlen (value), host_charset (), nullptr);
}

static inline PyObject *
to_python (std::string &&value)
{
  return PyUnicode_Decode (value.c_str (), value.size (),
			   host_charset (), nullptr);
}

static inline PyObject *
to_python (const std::string &value)
{
  return PyUnicode_Decode (value.c_str (), value.size (),
			   host_charset (), nullptr);
}

static inline PyObject *
to_python (gdb::unique_xmalloc_ptr<char> &&value)
{
  if (value == nullptr)
    return py_none ().release ();
  return PyUnicode_Decode (value.get (), strlen (value.get ()),
			   host_charset (), nullptr);
}

static inline PyObject *
to_python (gdbpy_ref<> &&value)
{
  return value.release ();
}

/* An instantiation of this function is used when calling a gdb method
   from Python.  It accepts some number of arguments (normally
   gdbpy_borrowed_ref or gdbpy_opt_borrowed_ref), and then calls the
   underlying function F.  Any exceptions are caught and converted,
   and the return value of F is converted to a Python object as
   appropriate.  */
template<auto F, typename... Args>
PyObject *
wrapped_function (Args... args)
{
  try
    {
      using result_type = typename std::invoke_result_t<decltype (F), Args...>;

      if constexpr (std::is_void_v<result_type>)
	{
	  F (args...);
	  return py_none ().release ();
	}
      else
	return to_python (F (args...));
    }
  catch (const gdb_python_exception &pye)
    {
      gdb_assert (PyErr_Occurred () != nullptr);
      return nullptr;
    }
  catch (const gdb_exception &exc)
    {
      return gdbpy_handle_gdb_exception (nullptr, exc);
    }
}

/* An instantiation of this function is used when calling a gdb method
   from Python.  It accepts some number of arguments (normally
   gdbpy_borrowed_ref or gdbpy_opt_borrowed_ref), and then calls the
   method METH.  Any exceptions are caught and converted, and the
   return value of METH is converted to a Python object as
   appropriate.  */
template<typename Class, typename Ret, typename... Args>
PyObject *
wrapped_method (Ret (Class::*meth) (Args...), Class *self, Args... args)
{
  try
    {
      if constexpr (std::is_void_v<Ret>)
	{
	  (self->*meth) (args...);
	  return py_none ().release ();
	}
      else
	return to_python ((self->*meth) (args...));
    }
  catch (const gdb_python_exception &pye)
    {
      gdb_assert (PyErr_Occurred () != nullptr);
      return nullptr;
    }
  catch (const gdb_exception &exc)
    {
      return gdbpy_handle_gdb_exception (nullptr, exc);
    }
}

/* A variant of wrapped_method that accepts a const method.  */
template<typename Class, typename Ret, typename... Args>
PyObject *
wrapped_method (Ret (Class::*meth) (Args...) const, Class *self, Args... args)
{
  try
    {
      if constexpr (std::is_void_v<Ret>)
	{
	  (self->*meth) (args...);
	  return py_none ().release ();
	}
      else
	return to_python ((self->*meth) (args...));
    }
  catch (const gdb_python_exception &pye)
    {
      gdb_assert (PyErr_Occurred () != nullptr);
      return nullptr;
    }
  catch (const gdb_exception &exc)
    {
      return gdbpy_handle_gdb_exception (nullptr, exc);
    }
}

template<auto F>
PyObject *
fn_wrapper (PyObject *self, PyObject *args, PyObject *kw)
{
  return wrapped_function<F> (gdbpy_borrowed_ref<> (args),
			      gdbpy_opt_borrowed_ref<> (kw));
}

template<typename C, auto M>
PyObject *
varargs_wrapper (PyObject *self, PyObject *args, PyObject *kw)
{
  return wrapped_method (M, static_cast<C *> (self),
			 gdbpy_borrowed_ref<> (args),
			 gdbpy_opt_borrowed_ref<> (kw));
}

} /* namespace safety_details */

/* Create a PyMethodDef for a no-argument method.  It takes the
   underlying class C and a pointer-to-method M as template
   parameters, and the name and documentation as arguments.  The
   method M is wrapped to call to_python and to catch exceptions per
   the safety protocol.  */
template<typename C, auto M>
constexpr PyMethodDef
noargs_method (const char *name, const char *doc)
{
  using namespace safety_details;
  return {
    name,
    [] (PyObject *self, PyObject *args) -> PyObject *
    {
      return wrapped_method (M, static_cast<C *> (self));
    },
    METH_NOARGS,
    doc,
  };
}

/* This is used to create the PyMethodDef for a varargs function.  It
   takes the underlying implementation function as a template
   argument, and also arguments for the method name and documentation
   string.

   The underlying function should accept a gdbpy_borrowed_ref argument
   (the arguments to the Python function), and then a
   gdbpy_opt_borrowed_ref for the keywords.  The function can return
   any type (see the to_python overloads); and should throw an
   exception on error.  If gdb_python_exception is thrown, the Python
   exception must already have been set.

   The gdb policy is that varargs methods must also accept keywords,
   and this is enforced here.  */
template<auto F>
constexpr PyMethodDef
varargs_function (const char *name, const char *doc)
{
  using namespace safety_details;
  return {
    name,
    (PyCFunction) fn_wrapper<F>,
    /* gdb's rule is that varargs should also use keywords.  */
    METH_VARARGS | METH_KEYWORDS,
    doc,
  };
}

/* Like a varargs function, but this implements a method on some
   Python type that gdb provides.  The implementation class and a
   pointer-to-method must be specified.  */
template<typename C, auto M>
constexpr PyMethodDef
varargs_method (const char *name, const char *doc)
{
  using namespace safety_details;
  return {
    name,
    (PyCFunction) varargs_wrapper<C, M>,
    /* gdb's rule is that varargs should also use keywords.  */
    METH_VARARGS | METH_KEYWORDS,
    doc,
  };
}

/* A function that wraps a "repr" or "str" method.  */
template<typename C, auto M>
PyObject *
wrap_tp_callback (PyObject *arg)
{
  using namespace safety_details;
  return wrapped_method (M, static_cast<C *> (arg));
}

/* A function that wraps a "get" method.  */
template<typename C, auto M>
PyObject *
wrap_getter (PyObject *arg, void *closure)
{
  using namespace safety_details;
  /* In gdb the closure argument is never used.  */
  return wrapped_method (M, static_cast<C *> (arg));
}

/* A function that wraps a "set" method.  */
template<typename C, void (C::*M) (gdbpy_opt_borrowed_ref<>)>
int
wrap_setter (PyObject *arg, PyObject *value, void *closure)
{
  using namespace safety_details;
  try
    {
      C *self = static_cast<C *> (arg);
      /* In gdb the closure argument is never used.  */
      (self->*M) (gdbpy_opt_borrowed_ref<> (value));
    }
  catch (const gdb_python_exception &pye)
    {
      gdb_assert (PyErr_Occurred () != nullptr);
      return -1;
    }
  catch (const gdb_exception &exc)
    {
      return gdbpy_handle_gdb_exception (-1, exc);
    }

  return 0;
}

#endif /* GDB_PYTHON_PY_SAFETY_H */
