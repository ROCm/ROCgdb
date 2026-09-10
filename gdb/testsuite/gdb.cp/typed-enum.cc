/* This testcase is part of GDB, the GNU debugger.

   Copyright 2020-2026 Free Software Foundation, Inc.

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

typedef unsigned char byte;

enum byte_enum : byte
{
  byte_val = 128
};

enum uchar_enum : unsigned char
{
  uchar_val = 128
};

enum int_enum : int
{
  int_three = 3,
  int_val = 128
};

enum schar_enum : signed char
{
  schar_neg = -128,
  schar_val = 127
};

int main()
{
  int v1 = byte_val;
  int v2 = uchar_val;
  int v3 = int_val;
  int v4 = schar_neg;
  return v1 == v2 && v3 == v4;
}
