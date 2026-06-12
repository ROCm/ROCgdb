/* Helpers related to Python object type

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

#include "python-internal.h"
#include "py-obj-type.h"

/* Return the type's fully qualified name from a PyTypeObject.  */
std::string
gdb_py_tp_name (PyTypeObject *py_type) noexcept
{
  static const std::string NO_TYPE_NAME = "<type name unavailable>";

  /* This helper should be used for cases when the called CPython function
     informs the caller that an error occurred, and a Python error was set.  */
  auto handle_err = [&]() -> std::string
  {
    gdbpy_print_stack ();
    return NO_TYPE_NAME;
  };

  /* Convert a PyObject to a UTF-8 encoded string.  */
  auto pyobj_to_str = [&](PyObject *name) -> std::string
  {
    const char *s = PyUnicode_AsUTF8AndSize (name, nullptr);
    if (s == nullptr)
      return handle_err ();
    return s;
  };

#if PY_VERSION_HEX >= 0x030d0000
  /* Notes:
     1. PyType_GetFullyQualifiedName() was added in version 3.13, and is
	part of the stable ABI since version 3.13.
     2. If an error occurs when looking up the module name (for instance,
	during the destruction of the object), PyType_GetFullyQualifiedName()
	returns NULL, and a Python error is set.  */
  gdbpy_ref<> fully_qualified_name (PyType_GetFullyQualifiedName (py_type));
  if (fully_qualified_name == nullptr)
    return handle_err ();
  return pyobj_to_str (fully_qualified_name.get ());

#else /* PY_VERSION_HEX < 0x030d0000 && ! defined (Py_LIMITED_API)  */
  /* For non-heap types, the fully qualified name corresponds to tp_name,
     which can never be NULL.  */
  if (! (PyType_GetFlags (py_type) & Py_TPFLAGS_HEAPTYPE))
    return py_type->tp_name;

  /* In the absence of PyType_GetFullyQualifiedName(), we fallback using
     __qualname__ instead. However, the result may differ slightly in some
     cases, e.g. the module name may be missing.  */

# if PY_VERSION_HEX >= 0x030b0000
  /* Notes:
     1. PyType_GetQualName() was added in version 3.11.
     2. On one hand, PyType_GetQualName() relies internally on ht_qualname
	which is supposed to never be NULL, therefore, does not set any Python
	error.  On the other hand, PyType_GetQualName() calls internally
	PyUnicode_AsUTF8AndSize(), which when erroring, sets a Python error
	and returns NULL.  */
  gdbpy_ref<> qualname (PyType_GetQualName (py_type));
  if (qualname == nullptr)
    return handle_err ();
  return pyobj_to_str (qualname.get ());

# else
  /* In the absence of PyType_GetQualName(), fallback on using PyHeapTypeObject
     which is not part of the public API.
     Tested on 3.10 which is the oldest supported version at the time of this
     writing, i.e. February 2026.  Hopefully, this workaround should go away
     when the minimum supported Python version is increased above 3.10.  */
  PyHeapTypeObject *ht = (PyHeapTypeObject *) py_type;
  if (ht->ht_qualname == nullptr)
    return NO_TYPE_NAME;
  return pyobj_to_str (ht->ht_qualname);
# endif
#endif
}

/* Return the type's fully qualified name from a PyObject.  */
std::string
gdbpy_py_obj_tp_name (PyObject *self) noexcept
{
  /* Note: Py_TYPE () is part of the stable ABI since version 3.14.  */
  return gdb_py_tp_name (Py_TYPE (self));
}
