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

/* Pointers to member functions of classes that use the multiple and the
   virtual inheritance model of the MSVC C++ ABI.  Every function
   returns a field of its own subobject, so a wrong "this" gives a wrong
   value.  The r_* globals hold what the program itself computed.  */

struct A { virtual int va (); int x = 11; };
int A::va () { return x; }
struct B { virtual int vb (); int y = 12; };
int B::vb () { return y; }
struct X { virtual int vx (); int xx = 13; };
int X::vx () { return xx; }
struct W2 { virtual int vw (); int w = 14; };
int W2::vw () { return w; }

/* Multiple inheritance: B at +16.  */
struct M2 : A, B { virtual int vm (); int nv (); int m = 21; };
int M2::vm () { return m; }
int M2::nv () { return m + 100; }

/* Virtual inheritance, own vfptr at +0, vbptr at +8.  */
struct V2 : virtual A { virtual int vv (); int z = 22; };
int V2::vv () { return z; }

/* Virtual inheritance, vbptr at +16 after a polymorphic base.  */
struct W3 : B, virtual A { int ww (); int w3 = 23; };
int W3::ww () { return w3; }

/* Virtual inheritance, the vbptr belongs to the base W5 at +16, so the
   vbtable's first entry is relative to W5, not to D4.  */
struct W5 : W2, virtual A { int w5f (); int w5 = 24; };
int W5::w5f () { return w5; }
struct D4 : X, W5 { int d4f (); int d4 = 25; };
int D4::d4f () { return d4; }

int (M2::*pm_a) () = &A::va;
int (M2::*pm_b) () = &B::vb;
int (M2::*pm_nv) () = &M2::nv;
int (M2::*pm_null) () = nullptr;
int (V2::*pv_own) () = &V2::vv;
int (V2::*pv_null) () = nullptr;
int (W3::*pw_b) () = &B::vb;
int (W3::*pw_own) () = &W3::ww;
int (D4::*pd_x) () = &X::vx;
int (D4::*pd_w5) () = &W5::w5f;
int (D4::*pd_own) () = &D4::d4f;

int r_pm_a, r_pm_b, r_pm_nv, r_pv_own, r_pw_b, r_pw_own;
int r_pd_x, r_pd_w5, r_pd_own;
bool r_pm_null, r_pv_null;

void
marker ()
{
}

int
main ()
{
  M2 m;
  V2 v;
  W3 w;
  D4 d;

  r_pm_a = (m.*pm_a) ();
  r_pm_b = (m.*pm_b) ();
  r_pm_nv = (m.*pm_nv) ();
  r_pv_own = (v.*pv_own) ();
  r_pw_b = (w.*pw_b) ();
  r_pw_own = (w.*pw_own) ();
  r_pd_x = (d.*pd_x) ();
  r_pd_w5 = (d.*pd_w5) ();
  r_pd_own = (d.*pd_own) ();
  r_pm_null = (pm_null == nullptr);
  r_pv_null = (pv_null == nullptr);

  marker ();		/* Stop here.  */
  return 0;
}
