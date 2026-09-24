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

volatile int global_var = 0;

static inline void __attribute__ ((__always_inline__))
inline_func (void)
{
  ++global_var;		/* Breakpoint here.  */
}

int __attribute__ ((noinline, noclone))
foo (int x)
{
  inline_func ();
  return x + global_var;
}

int __attribute__ ((noinline, noclone))
bar (int x)
{
  inline_func ();
  return x - global_var;
}

int
main (void)
{
  ++global_var;

  int ans = foo (42) + bar (10);

  ++global_var;

  return ans - global_var;
}
