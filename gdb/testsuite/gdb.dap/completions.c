/* Copyright 2026 Free Software Foundation, Inc.

   This file is part of GDB.

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

int unique_enough_prefix_1 = 23;

int unique_enough_prefix_2 = 91;

int inner_func ()
{
  int unique_enough_prefix_4 = 19;
  return unique_enough_prefix_4 - 19; /* BREAK */
}

int main ()
{
  int unique_enough_prefix_3 = 17;
  return inner_func () + unique_enough_prefix_3 - 17;
}
