/* Type traits relating to GDB's Python integration.

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

#ifndef GDB_PYTHON_PYTHON_TRAITS_H
#define GDB_PYTHON_PYTHON_TRAITS_H

namespace gdb
{
/* All of our custom Python types are created as structs, like this:

   struct some_new_type : public PyObject
   {
     ... various fields ...
   };

   Then instances of this struct are created by calling PyObject_New,
   either directly within GDB's C++ code, or within Python when a user's
   Python script creates an instance of that class.

   The problem is that Python is written in C, and PyObject_New doesn't
   call any constructors for `some_new_type`, nor for any of the fields
   within `some_new_type`.

   If `some_new_type` is Plain Old Data (POD), then this is fine.  Or, to
   be more C++ specific, if `some_new_type` is trivially default
   constructable, then we're fine.

   But if a field within `some_new_type` has a non-trivial constructor,
   then we're in trouble as that constructor will never be run.

   An example of a problematic field type is frame_info_ptr.  The
   constructor for this type registers the new object with a central
   management object, recording the `this` pointer, using this type within
   `some_new_type` will not work as expected; frame invalidation will not
   show up within the frame_info_ptr as you might expect.

   And so, this type trait exists.  Whenever a struct is created to define
   a new Python type we should add a line like:

     static_assert (gdb::is_python_allocatable_v<some_new_type>);

   This will fail if any field of `some_new_type` is unsuitable for this
   use.

   We don't actually check is_trivially_default_constructible here.  Some
   types, e.g. ui_file_style::color, have non-trivial (or no default)
   constructors, but are still safe to use within `some_new_type` because
   their constructors just initialise data fields; there's nothing
   "special" that the constructor does that cannot be achieved by
   assigning the fields after creation with PyObject_New.

   What actually matters is that the type is trivially destructible
   (Python won't call C++ destructors, so destructors with side effects,
   like deregistering from a list, would be skipped) and trivially
   copyable (Python may copy objects with memcpy).  Types like
   frame_info_ptr, whose constructors and destructors have side effects
   such as registering with a central management object, will be caught
   because they are neither trivially destructible nor trivially copyable.
   Types like ui_file_style are trivially destructible and copyable, so
   pass this trait.  */

template<typename T>
struct is_python_allocatable
{
  static constexpr bool value =
    std::is_trivially_destructible_v<T>
    && std::is_trivially_copyable_v<T>;
};

/* Helper for the above trait to make it more usable.  */

template<typename T>
inline constexpr bool is_python_allocatable_v
  = is_python_allocatable<T>::value;
}

#endif /* GDB_PYTHON_PYTHON_TRAITS_H */
