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

/* Virtual calls that reach the final overrider through a compiler
   generated this-adjusting thunk.  */

/* M::f overrides B::f, but B is not M's first base, so a call through
   a B pointer goes through a thunk that adjusts "this" from B to M.
   The padding makes the adjustment large.  */

struct A
{
  virtual int f ();
  char pad[300];
};

struct B
{
  virtual int f ();
  int b = 2;
};

struct M : A, B
{
  int f () override;
  int m = 3;
};

int A::f () { return 0; }
int B::f () { return b; }

int
M::f ()
{
  return m;			/* M::f body.  */
}

/* W::g overrides a function of its virtual base V and W's constructor
   calls it virtually, so under the MSVC C++ ABI the call goes through
   a "vtordisp" thunk that reads the this-adjustment from the object.  */

struct V
{
  virtual int g ();
  int v = 4;
};

struct W : virtual V
{
  W ();
  int g () override;
  int w = 5;
};

struct X : W
{
  X () {}
  int x = 6;
};

int V::g () { return v; }

W::W ()
{
  V *vp = this;
  vp->g ();			/* Call from W's constructor.  */
}

int
W::g ()
{
  return w;			/* W::g body.  */
}

int
main ()
{
  M m;
  X x;
  B *bp = &m;
  V *vp = &x;
  int r = bp->f ();		/* Call through B.  */
  r += vp->g ();		/* Call through V.  */
  return r == 0;
}
