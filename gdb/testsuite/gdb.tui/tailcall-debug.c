/* Copyright 2026 Free Software Foundation, Inc.

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

void
callee (void)
{
  asm ("callee_label: .globl callee_label");
  ++global_var;
}

/* When we generate the DWARF with the DWARF assembler, we split caller in
   half.  Everything up to the 'callee ();' call is left as caller, and
   everything after that becomes dummy_func.  */

void
caller (void)
{
  asm ("caller_label: .globl caller_label");
  callee ();
  ++global_var;
  ++global_var;
}

int
main (void)
{
  asm ("main_label: .globl main_label");
  caller ();
  return 0;
}
