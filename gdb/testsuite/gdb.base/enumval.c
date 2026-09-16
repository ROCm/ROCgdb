/* This testcase is part of GDB, the GNU debugger.

   Copyright 2012-2026 Free Software Foundation, Inc.

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

enum wide_values { I, J = 0xffffffffU, K = 0xf000000000000000ULL }
  e = J, f = K;

/* Enum that mixes a negative enumerator with one that could be misinterpreted
   if the compiler describes its value with DW_FORM_data1 (see PR
   symtab/34616).  */

enum mixed { M_NEG = -1, M = 200 } g = M;

/* The same, but with no negative enumerator, so that the enum is
   unsigned.  */

enum unmixed { U_ZERO, U = 200 } h = U;

enum { ZERO };

void
dummy()
{
}

int
main(void)
{
  dummy();
  return ZERO; /* This is here to ensure it survives into the debug info.  */
}
