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

/* A policy class for gdb::ref_ptr for Python reference counting.  */
template<typename T>
struct gdbpy_ref_policy
{
  static_assert(std::is_base_of<PyObject, T>::value,
		"T must be a subclass of PyObject");

  static void incref (T *ptr)
  {
    Py_INCREF (static_cast<PyObject *> (ptr));
  }

  static void decref (T *ptr)
  {
    Py_DECREF (static_cast<PyObject *> (ptr));
  }
};

/* A gdb::ref_ptr that has been specialized for Python objects or
   their "subclasses".  */
template<typename T = PyObject> using gdbpy_ref
  = gdb::ref_ptr<T, gdbpy_ref_policy<T>>;

/* A wrapper class for Python extension objects that have a __dict__ attribute.

   Any Python C object extension needing __dict__ should inherit from this
   class. Given that the C extension object must also be convertible to
   PyObject, this wrapper class publicly inherits from PyObject as well.

   Access to the dict requires a custom getter defined via PyGetSetDef.
     gdb_PyGetSetDef my_object_getset[] =
     {
       { "__dict__", gdb_py_generic_dict_getter, nullptr,
	 "The __dict__ for this object.", nullptr },
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
};

#endif /* GDB_PYTHON_PY_REF_H */
