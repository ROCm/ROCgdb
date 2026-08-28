/* PDB Test Program — exercises many CodeView symbol and type records.

   Compile with clang-cl:
     clang-cl /Zi /Od /Fe:pdb-test.exe pdb-test.cpp

   Alternative (clang++):
     clang++ -g -gcodeview -fuse-ld=lld -o pdb-test.exe pdb-test.cpp

   This test program is shared across gdb.pdb test scripts.

   Copyright (C) 2026 Free Software Foundation, Inc.

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

#include <cstdio>

/* Compiler-specific inlining keywords */
#ifndef __forceinline
#ifdef _MSC_VER
#define __forceinline __forceinline
#else
#define __forceinline inline __attribute__((always_inline))
#endif
#endif

/* ---- Global variables (S_GDATA32) ---- */
int g_counter = 0;
const int g_max_iterations = 100;
volatile int g_flag = 1;
char g_tag = 'A';
int g_array[5] = {10, 20, 30, 40, 50};
int *g_ptr = &g_counter;

/* ---- File-static variable (S_LDATA32) ---- */
static int s_accumulator = 0;

/* ---- Enum (LF_ENUM + S_CONSTANT for enumerators) ---- */
enum Color { Red = 0, Green = 1, Blue = 2 };

static Color s_color = Red;

/* ---- Struct (LF_STRUCTURE) ---- */
struct Point
{
  int x;
  int y;
};

/* ---- Class with virtual method (LF_CLASS, LF_MFUNCTION, S_THUNK32) ---- */
class Shape
{
public:
  unsigned flags : 4;   /* LF_BITFIELD */
  unsigned id    : 12;

  Shape () : flags (0), id (0) {}
  virtual ~Shape () {}
  virtual double area () const { return 0.0; }
  int get_id () const { return id; }
};

/* ---- Derived class — virtual override ---- */
class Circle : public Shape
{
  double radius_;

public:
  explicit Circle (double r) : radius_ (r) { flags = 1; }
  double area () const override { return 3.14159265 * radius_ * radius_; }
  double get_radius () const { return radius_; }
};

/* ---- Union (LF_UNION) ---- */
union Value
{
  int   i;
  float f;
  char  c[4];
};

/* ---- Typedef / using alias (S_UDT) ---- */
using PointPtr = Point *;

/* ---- Function-pointer type (LF_PROCEDURE) ---- */
typedef int (*BinaryOp) (int, int);

/* ---- Anonymous-namespace function (S_LPROC32 — internal linkage) ---- */
namespace {
int
internal_helper (int val)
{
  return val * 3 + 1;
}
} /* anonymous namespace */

/* ---- Inline function (S_INLINESITE when inlined) ---- */
__forceinline int
fast_abs (int x)
{
  return x < 0 ? -x : x;
}

/* ---- Functions with const / volatile / reference params ---- */
int
add_values (int a, int b)
{
  return a + b;
}

/* ---- Also define a function with const ref params for type testing ---- */
int
sum_with_refs (const int &a, const int &b)
{
  return a + b;
}

void
modify_volatile (volatile int *p, int val)
{
  *p = val;
}

int
process_point (const Point &pt)
{
  return pt.x + pt.y;
}

/* ---- 3rd-level helper (file-static, nested blocks + S_BLOCK32) ---- */
static int
static_helper (int x, int y)
{
  int result = 0;
  {
    int temp = x * 2;
    {
      int inner = temp + y;
      result = inner;
    }
  }
  return result;
}

/* Debugger set-variable verification target.  The test breaks at
   set_marker(), modifies local/global/static variables in main's frame,
   then continues to verify_marker() and checks the parameters to confirm
   the inferior actually used the modified values.  */
static void set_marker () { }

static int
verify_marker (int local_val, int global_val, int static_val, int flag_val)
{
  return local_val + global_val + static_val + flag_val;
}

/* ---- 2nd-level compute (goto label -> S_LABEL32) ---- */
int
compute1 (int a, int b, BinaryOp op)
{
  int partial = op (a, b);
  int adjusted = static_helper (partial, b);

  if (adjusted < 0)
    goto negative;

  return adjusted;

negative:
  return -adjusted;
}

/* ---- Main entry point ---- */
int
main ()
{
  /* Local variables of various types (S_LOCAL32, S_REGREL32).  */
  volatile int local_int = 42;
  double local_double = 3.14;
  Color l_color = Green;
  Point origin = {0, 0};
  PointPtr pp = &origin;

  Circle circle (5.0);
  Shape *shape = &circle;

  volatile int *another_int = &local_int;

  Value val;
  val.i = 0x41424344;

  /* Function pointer.  */
  BinaryOp adder = add_values;

  /* Short iteration — 3 passes.  */
  for (int i = 0; i < 3; i++)
    {
      int step = internal_helper (i);
      s_accumulator += step;
      g_counter++;

      s_color = Blue;

      /* Arithmetic + bitwise.  */
      int math = (step * 2 + i - 1) % 7;
      math = math | (i << 2);
      math = fast_abs (math);
      s_accumulator += math;
    }

  /* Exercise various constructs.  */
  int sum = compute1 (local_int, 10, adder);
  double circ_area = shape->area () - (float)(*another_int);
  volatile int squared = (*another_int) * (*another_int);
  int pt_val = process_point (origin);

  /* Debugger set-variable test: test modifies variables at set_marker,
     then verify_marker receives and returns the modified values.  */
  set_marker ();
  (void) verify_marker (local_int, g_counter, s_accumulator, g_flag);

  modify_volatile (&g_flag, 0);

  /* Use local_double to prevent unused-variable warning.  */
  if (local_double > 0.0)
    g_tag = 'B';

  l_color = Red;

  /* Prevent optimizer from removing everything.  */
  printf ("sum=%d area=%.2f squired= %d, pt=%d acc=%d val=%c tag=%c flag=%d s_color:%d  %d\n",
	  sum, circ_area, squared, pt_val, s_accumulator, val.c[0], g_tag, g_flag, s_color, l_color);

  (void) pp;

  return 0;
}
