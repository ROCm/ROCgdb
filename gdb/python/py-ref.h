/* Python reference-holding class

   Copyright (C) 2016-2026 Free Software Foundation, Inc.

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

#ifndef GDB_PYTHON_PY_REF_H
#define GDB_PYTHON_PY_REF_H

#include "gdbsupport/gdb_ref_ptr.h"
#include "gdbsupport/traits.h"
#include "python-traits.h"

/* A policy class for gdb::ref_ptr for Python reference counting.  */
struct gdbpy_ref_policy
{
  static void incref (PyObject *ptr)
  {
    Py_INCREF (ptr);
  }

  static void decref (PyObject *ptr)
  {
    Py_DECREF (ptr);
  }
};

/* A gdb::ref_ptr that has been specialized for Python objects or
   their "subclasses".  */
template<typename T = PyObject> using gdbpy_ref
  = gdb::ref_ptr<T, gdbpy_ref_policy>;

/* A class representing an optional borrowed reference.  It is
   "optional" because NULL is a valid value.

   This is a simple wrapper for a pointer to PyObject or some subclass
   of it.  Aside from documenting what the code does, the main
   advantage of using this is that conversion to a gdbpy_ref<T> is
   prevented.

   An optional borrowed reference is only used in situations where
   Python says NULL is valid.  For example, it is used as the type of
   the "keywords" argument to a varargs method.  Most code should
   prefer an ordinary gdbpy_borrowed_ref, see below.  */
template<typename T = PyObject>
class gdbpy_opt_borrowed_ref
{
public:

  gdbpy_opt_borrowed_ref (T *obj)
    : m_obj (obj)
  {
  }

  template<typename U>
  gdbpy_opt_borrowed_ref (const gdbpy_ref<U> &ref)
    : m_obj (ref.get ())
  {
  }

  operator T * () const
  {
    return m_obj;
  }

  operator gdbpy_ref<T> () = delete;

protected:

  T *m_obj;
};

/* A borrowed reference that is guaranteed not to be NULL.

   Like gdbpy_opt_borrowed_ref, this mostly serves a documentary
   purpose.  However, it also allows a checked cast to any subclass of
   T, and conversion to a gdbpy_ref<T> will automatically acquire a
   new reference -- a safety improvement over plain PyObject * or the
   like.  */
template<typename T = PyObject>
class gdbpy_borrowed_ref : public gdbpy_opt_borrowed_ref<T>
{
public:

  gdbpy_borrowed_ref (T *obj)
    : gdbpy_opt_borrowed_ref<T> (obj)
  {
    gdb_assert (this->m_obj != nullptr);
  }

  template<typename U>
  gdbpy_borrowed_ref (const gdbpy_ref<U> &ref)
    : gdbpy_opt_borrowed_ref<T> (ref)
  {
    gdb_assert (this->m_obj != nullptr);
  }

  gdbpy_borrowed_ref (std::nullptr_t) = delete;

  /* Allow a (checked) conversion to any subclass of T.  */
  template<typename U,
	   typename = gdb::Requires<std::is_convertible<U *, T *>>>
  operator U * () const
  {
    gdb_assert (PyObject_TypeCheck (this->m_obj,
				    U::corresponding_object_type));
    return static_cast<U *> (this->m_obj);
  }

  /* When converting a borrowed reference to a gdbpy_ref<>, a new
     reference is acquired.  */
  operator gdbpy_ref<T> () const
  {
    return gdbpy_ref<T>::new_reference (this->m_obj);
  }
};

/* A wrapper class for Python extension objects that have a __dict__ attribute.

   Any Python C object extension needing __dict__ should inherit from this
   class. Given that the C extension object must also be convertible to
   PyObject, this wrapper class publicly inherits from PyObject as well.

   Access to the dict requires a custom getter defined via PyGetSetDef.
     gdb_PyGetSetDef my_object_getset[] =
     {
       gdbpy_dict_wrapper_cfg_dict_getter ("object"),
       ...
       { nullptr }
     };

   It is also important to note that __dict__ is used during the attribute
   look-up. Since this dictionary is not managed by Python and is not exposed
   via tp_dictoffset, custom attribute getter (tp_getattro) and setter
   (tp_setattro) are required to correctly redirect attribute access to the
   dictionary:
     - gdb_py_generic_getattro (), assigned to tp_getattro for static types,
       or Py_tp_getattro for heap-allocated types.
     - gdb_py_generic_setattro (), assigned to tp_setattro for static types,
       or Py_tp_setattro for heap-allocated types.  */
struct gdbpy_dict_wrapper : public PyObject
{
  /* Dictionary holding user-added attributes.
     This is the __dict__ attribute of the object.  */
  PyObject *dict;

  /* Compute the address of the __dict__ attribute for the given PyObject.  */
  static PyObject **compute_addr (PyObject *self)
  {
    auto *wrapper = reinterpret_cast<gdbpy_dict_wrapper *> (self);
    return &wrapper->dict;
  }

#define gdbpy_dict_wrapper_cfg_dict_getter(object_name)	\
  {							\
    "__dict__", /* name */				\
    (getter) gdb_py_generic_dict_getter,		\
    (setter) nullptr,					\
    "The __dict__ for this " object_name ".", /* doc */	\
    nullptr, /* closure */				\
  }

#define gdbpy_dict_wrapper_getsetattro	\
  /*tp_getattro*/			\
  gdb_py_generic_getattro,		\
  /*tp_setattro*/			\
  gdb_py_generic_setattro

  /* Allocate the dictionary pointed by 'dict'.
     Note: this method should be called once the object was allocated,
     when setting its attributes.  */
  bool allocate_dict ()
  {
    dict = PyDict_New ();
    return dict != nullptr;
  }
};

static_assert (gdb::is_python_allocatable_v<gdbpy_dict_wrapper>);

#endif /* GDB_PYTHON_PY_REF_H */
