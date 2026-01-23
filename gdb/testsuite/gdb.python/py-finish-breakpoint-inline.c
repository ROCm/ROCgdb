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
   along with this program.  If not, see  <http://www.gnu.org/licenses/>.  */

/* Global used to create filler code within functions.  */
volatile int global_var = 1;

static int  __attribute__ ((noinline, noclone))
baz (int arg)
{
  arg += global_var;
  return arg;
}

static inline int __attribute__ ((__always_inline__))
bar (int arg)
{
  arg += global_var;
  arg = baz (arg);		/* Finish location.  */
  arg -= global_var;
  return arg;
}

static inline int __attribute__ ((__always_inline__))
foo (int arg)
{
  arg += global_var;
  arg = bar (arg);
  arg -= global_var;
  return arg;
}

int
main (void)
{
  int ans;

  ++global_var;
  ++global_var;
  ans = foo (42);
  ++global_var;
  ++global_var;
  ans += global_var;
  return ans;		/* Final breakpoint.  */
}
