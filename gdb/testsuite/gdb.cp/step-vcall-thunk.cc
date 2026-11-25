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

/* Calls through pointers to virtual member functions.  Under the MSVC
   C++ ABI such a pointer holds the address of a compiler-generated
   "vcall" thunk that dispatches through the object's vtable.  */

struct S
{
  virtual int f ();
  virtual int g ();
  int s = 1;
};

struct D : S
{
  int g () override;
  int d = 2;
};

int S::f () { return 0; }
int S::g () { return s; }

int
D::g ()
{
  return d;			/* D::g body.  */
}

/* M::h overrides both A::h and B::h.  It expects "this" to point to
   the A part of M, so the vtable slot reached through B holds a
   this-adjusting thunk.  */

struct A
{
  virtual int h ();
  char pad[64];
};

struct B
{
  virtual int h ();
  int b = 3;
};

struct M : A, B
{
  int h () override;
  int m = 4;
};

int A::h () { return 0; }
int B::h () { return b; }

int
M::h ()
{
  return m;			/* M::h body.  */
}

/* L::v17 is in a vtable slot far enough from the start that the
   thunk addresses it with a 32-bit displacement.  */

struct L
{
  virtual int v0 (); virtual int v1 (); virtual int v2 ();
  virtual int v3 (); virtual int v4 (); virtual int v5 ();
  virtual int v6 (); virtual int v7 (); virtual int v8 ();
  virtual int v9 (); virtual int v10 (); virtual int v11 ();
  virtual int v12 (); virtual int v13 (); virtual int v14 ();
  virtual int v15 (); virtual int v16 (); virtual int v17 ();
  int l = 5;
};

int L::v0 () { return 0; }
int L::v1 () { return 1; }
int L::v2 () { return 2; }
int L::v3 () { return 3; }
int L::v4 () { return 4; }
int L::v5 () { return 5; }
int L::v6 () { return 6; }
int L::v7 () { return 7; }
int L::v8 () { return 8; }
int L::v9 () { return 9; }
int L::v10 () { return 10; }
int L::v11 () { return 11; }
int L::v12 () { return 12; }
int L::v13 () { return 13; }
int L::v14 () { return 14; }
int L::v15 () { return 15; }
int L::v16 () { return 16; }

int
L::v17 ()
{
  return l;			/* L::v17 body.  */
}

int
call_s (S *p, int (S::*pm) ())
{
  return (p->*pm) ();		/* Call through S.  */
}

int
call_b (B *p, int (B::*pm) ())
{
  return (p->*pm) ();		/* Call through B.  */
}

int
call_l (L *p, int (L::*pm) ())
{
  return (p->*pm) ();		/* Call through L.  */
}

int
main ()
{
  D d;
  M m;
  L l;
  int r = call_s (&d, &S::g);
  r += call_b (&m, &B::h);
  r += call_l (&l, &L::v17);
  return r == 0;
}
