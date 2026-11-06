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

#include <typeinfo>

/* Two polymorphic bases: the second base is a subobject at a non-zero
   offset, with its own vfptr.  */

struct A
{
  virtual int fa () { return 1; }
  int a = 1;
};

struct B
{
  virtual int fb () { return 2; }
  int b = 2;
};

struct M : A, B
{
  int fb () override { return 3; }
  int m = 3;
};

/* A polymorphic virtual base.  X declares no virtual function of its
   own, so its only vfptr is the one in V.  */

struct V
{
  virtual int fv () { return 4; }
  int v = 4;
};

struct X : virtual V
{
  int x = 5;
};

/* Y overrides a function of its virtual base and has a constructor, so
   the V subobject gets a vtordisp.  Z puts Y at a non-zero offset,
   which moves V while Y's constructor runs.  */

struct Y;
void marker_ctor (Y *yp) { }

struct Y : virtual V
{
  Y () { marker_ctor (this); }
  int fv () override { return 6; }
  int y = 6;
};

struct P
{
  int p[3] = { 7, 8, 9 };
};

struct Z : P, Y
{
  int z = 10;
};

namespace ns
{
  struct N : A
  {
    int n = 11;
  };
}

/* Objects of the base classes themselves, so that the compiler emits
   their full debug info.  */
A a_obj;
B b_obj;
V v_obj;

M m;
X x;
ns::N n;

B *bp = &m;
V *xvp = &x;
A *nap = &n;

int iarr[2][3];
const A *cap;

int
main ()
{
  Z z;
  V *zvp = &z;

  const std::type_info &ti_m = typeid (*bp);
  const std::type_info &ti_x = typeid (*xvp);
  const std::type_info &ti_n = typeid (*nap);
  const std::type_info &ti_iarr = typeid (iarr);
  const std::type_info &ti_cap = typeid (cap);

  return zvp->fv () + bp->fb (); /* break here */
}
