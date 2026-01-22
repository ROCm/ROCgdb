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

/* This test relies on tailcall_function actually compiling to a tail call
   function.  we try to force this by preventing tailcall_function and
   normal_function from being inlined, then compiling this file at -O2.
   Still, that's no guarantee.  If tailcall_function isn't a tail call,
   then the test becomes pointless.  */

volatile int global_var = 42;

int __attribute__ ((noinline, noclone))
normal_function (int x)
{
  return x + 1;
}

int __attribute__ ((noinline, noclone))
tailcall_function (int x)
{
  ++global_var;
  return normal_function (x);
}

int
main (void)
{
  int result = tailcall_function (42);
  result -= global_var;			/* Temporary breakpoint here.  */
  return result;
}
