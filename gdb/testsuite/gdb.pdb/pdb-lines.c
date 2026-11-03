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

/* Three functions, each its own section contribution when built with
   /Gy.  */

__declspec (noinline) int
first (int x)
{
  return x + 1;		/* first body */
}

__declspec (noinline) int
second (int x)
{
  return x * 2;		/* second body */
}

int
main (void)
{
  return first (1) + second (2);	/* main body */
}
