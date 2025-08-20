/* Copyright (C) 2008-2026 Free Software Foundation, Inc.

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

/* Force outputs to unbuffered mode.  */

#include <stdio.h>

/* Use an explicit priority so that this runs before constructors of
   namespace-scope C++ objects (which may output to stdout/stderr).
   Lower priorities run first.  Constructor priorities from 0 to 100
   are reserved for the implementation.  */
static void __attribute__ ((constructor (101)))
__gdb_set_unbuffered_output (void)
{
  setvbuf (stdout, NULL, _IONBF, BUFSIZ);
  setvbuf (stderr, NULL, _IONBF, BUFSIZ);
}
