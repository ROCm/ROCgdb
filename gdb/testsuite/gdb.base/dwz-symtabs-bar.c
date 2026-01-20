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

#include "dwz-symtabs-bar.h"
#include "dwz-symtabs-common.h"

static int bar_data = 0;

void
process_bar_data (int value)
{
  bar_data += add_some_int (3, value);

}

volatile int *ptr = 0;

void bar_func (void)
{
  /* This will hopefully trigger a segfault.  */
  *ptr = 0;	/* Crash here.  */
}
