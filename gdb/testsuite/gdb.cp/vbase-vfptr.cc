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

/* Classes with a virtual base.  W also declares a virtual function, so
   under the MSVC C++ ABI it has its own vfptr, which comes before the
   vbptr.  Q has a vfptr too, but its virtual base comes through P, a
   non-virtual base with its own vbptr.  */

struct V
{
  int v = 11;
};

struct W : virtual V
{
  virtual int n ();
  int w = 22;
};

struct P : virtual V
{
  int p = 33;
};

struct Q : P
{
  virtual int q ();
  int qq = 44;
};

int W::n () { return w; }
int Q::q () { return qq; }

W gw;
Q gq;

int
main ()
{
  V *a = &gw;
  V *b = &gq;
  return a->v + b->v;	/* Break here.  */
}
