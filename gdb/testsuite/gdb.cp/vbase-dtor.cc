/* This testcase is part of GDB, the GNU debugger.

   Copyright 2026 Free Software Foundation, Inc.

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

/* D has a shared virtual base V.  Destroying a complete D object must
   run D's own destructor body and then destroy V exactly once.  Under
   the MSVC C++ ABI the compiler emits this as a separate "vbase
   destructor" (mangled "??_D"), which calls D's "??1" destructor and
   then V's.  Under the Itanium ABI it is the complete-object
   destructor (D1).  */

int v_dtor_count = 0;

struct V
{
  int v = 1;

  ~V ()
  {
    v_dtor_count++;	/* V dtor.  */
  }
};

struct A : virtual V
{
  int a = 2;

  ~A ()
  {
  }
};

struct B : virtual V
{
  int b = 3;

  ~B ()
  {
  }
};

struct D : A, B
{
  int d = 4;

  ~D ()
  {
    d = 0;	/* D dtor.  */
  }
};

int
main ()
{
  D *p = new D;

  delete p;	/* Delete D.  */

  return v_dtor_count;	/* After delete.  */
}
