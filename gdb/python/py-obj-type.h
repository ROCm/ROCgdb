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

#ifndef GDB_PYTHON_PY_OBJ_TYPE_H
#define GDB_PYTHON_PY_OBJ_TYPE_H

/* Return the type's fully qualified name from a PyTypeObject.  */
extern const char *gdb_py_tp_name (PyTypeObject *py_type) noexcept;

/* Return the type's fully qualified name from a PyObject.  */
extern const char *gdbpy_py_obj_tp_name (PyObject *self) noexcept;

#endif /* GDB_PYTHON_PY_OBJ_TYPE_H */
