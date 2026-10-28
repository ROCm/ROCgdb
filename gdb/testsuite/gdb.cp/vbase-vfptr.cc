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
   non-virtual base with its own vbptr.  F and H have a virtual base of
   their own and reuse the vbptr of a non-virtual base that is not at
   offset 0: in F that base comes after X; in H, P is declared first but
   W, which has a vfptr, is laid out first.  */

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

struct X
{
  virtual int x ();
  int xx = 55;
};

struct B
{
  int b = 66;
};

struct F : X, W, virtual B
{
  int f = 77;
};

struct H : P, W, virtual B
{
  int h = 88;
};

int W::n () { return w; }
int Q::q () { return qq; }
int X::x () { return xx; }

W gw;
Q gq;
F gf;
H gh;

int
main ()
{
  V *a = &gw;
  V *b = &gq;
  return a->v + b->v;	/* Break here.  */
}
