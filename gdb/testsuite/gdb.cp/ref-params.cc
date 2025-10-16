/* This test script is part of GDB, the GNU debugger.

   Copyright 2006-2026 Free Software Foundation, Inc.

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

/* Author: Paul N. Hilfinger, AdaCore Inc. */

struct Parent {
  Parent (int id0) : id(id0) { }
  int id;
};

struct Child : public Parent {
  Child (int id0) : Parent(id0) { }
};

int f1(Parent& R)
{
  return R.id;			/* Set breakpoint marker3 here.  */
}

int f2(Child& C)
{
  return f1(C);			/* Set breakpoint marker2 here.  */
}

int f3(const int &i)
{
  return i;
}

struct OtherParent {
  OtherParent (int other_id0) : other_id(other_id0) { }
  int other_id;
};

struct MultiChild : public Parent, OtherParent {
  MultiChild (int id0) : Parent(id0), OtherParent(id0 * 2) { }
};

int mf1(OtherParent& R)
{
  return R.other_id;
}

int mf2(MultiChild& C)
{
  return mf1(C);
}

/* A class to be used by functions and methods exercising c++/25957.  */
class TestClass
{
  public:
  int value;

  TestClass (int v) : value (v) {}

  int
  const_ref_method (const int &x) const
  {
    return value + x;
  }

  int
  const_ref_method (const TestClass &obj) const
  {
    return value + obj.value;
  }

  int
  ref_method (int &x)
  {
    return value + 2 * x;
  }

  int
  ref_method (TestClass &obj)
  {
    return value + 2 * obj.value;
  }
};

/* Define globals to be used by functions and methods exercising c++/25957.  */
int global_int = 42;
TestClass global_obj (10);

/* Helper functions to test reference parameter behavior for c++/25957.  */
int
const_ref_func (const int &x)
{
  return x * 2;
}

int
const_ref_func (const TestClass &obj)
{
  return obj.value * 2;
}

int
ref_func (int &x)
{
  return x + 1;
}

int
ref_func (TestClass &obj)
{
  return obj.value + 1;
}

int main(void)
{
  Child Q(42);
  Child& QR = Q;

  /* Set breakpoint marker1 here.  */

  f2(Q);
  f2(QR);
  f3(Q.id);

  MultiChild MQ(53);
  MultiChild& MQR = MQ;

  mf2(MQ);			/* Set breakpoint MQ here.  */

  TestClass obj (5);
  int local_var = 15;

  /* Prevent compiler from optimizing away the function and method calls.  */
  int dummy_int = 99;
  (void) const_ref_func (dummy_int);
  (void) const_ref_func (global_int);
  (void) const_ref_func (obj);
  (void) const_ref_func (global_obj);
  (void) ref_func (dummy_int);
  (void) ref_func (global_int);
  (void) ref_func (obj);
  (void) ref_func (global_obj);
  (void) obj.const_ref_method (dummy_int);
  (void) obj.const_ref_method (global_int);
  (void) obj.const_ref_method (obj);
  (void) obj.const_ref_method (global_obj);
  (void) obj.ref_method (dummy_int);
  (void) obj.ref_method (global_int);
  (void) obj.ref_method (obj);
  (void) obj.ref_method (global_obj);
  (void) global_obj.const_ref_method (dummy_int);
  (void) global_obj.const_ref_method (global_int);
  (void) global_obj.const_ref_method (obj);
  (void) global_obj.const_ref_method (global_obj);
  (void) global_obj.ref_method (dummy_int);
  (void) global_obj.ref_method (global_int);
  (void) global_obj.ref_method (obj);
  (void) global_obj.ref_method (global_obj);

  /* Breakpoint here for c++/25957 testing.  */
  return 0;  /* breakpoint-here */
}
