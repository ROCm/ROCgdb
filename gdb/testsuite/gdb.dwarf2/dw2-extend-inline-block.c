/* This testcase is part of GDB, the GNU debugger.

   Copyright 2025 Free Software Foundation, Inc.

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

/* The follow code exists only to be referenced from the generated line
   table.  */
#if 0
static inline void
foo (void)
{
  /* foo:1 */
  /* foo:2 */
  /* foo:3 */
}

int
main (void)
{						/* main decl line */
  /* main:1 */
  /* main:2 */
  /* main:3 */ foo ();				/* foo call line */
  /* main:4 */
  /* main:5 */
  /* main:6 */
}
#endif


int
main (void)
{
  asm ("main_label: .globl main_label");
  ++global_var;

  asm ("main_0: .globl main_0");
  ++global_var;

  asm ("main_1: .globl main_1");
  ++global_var;

  asm ("main_2: .globl main_2");
  ++global_var;

  asm ("main_3: .globl main_3");
  ++global_var;

  asm ("main_4: .globl main_4");
  ++global_var;

  asm ("main_5: .globl main_5");
  ++global_var;

  asm ("main_6: .globl main_6");
  ++global_var;

  asm ("main_7: .globl main_7");
  ++global_var;

  asm ("main_8: .globl main_8");
  ++global_var;

  return 0;
}
