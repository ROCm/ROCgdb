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


typedef unsigned short pkg__myint;

typedef struct {
  pkg__myint F;
}  __attribute__ ((aligned (8))) pkg__myint___PAD;

pkg__myint
call (pkg__myint___PAD padded)
{
  return padded.F + 7;
}

pkg__myint value = 16;
pkg__myint___PAD pdv = { 16 };

int
main()
{
  return 0;
}
