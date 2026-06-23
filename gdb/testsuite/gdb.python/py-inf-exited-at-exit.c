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

#include <unistd.h>
#include "gdb_watchdog.h"

/* GDB can set GLOBAL_VAR to non-zero to cause the inferior to exit.  */
volatile int global_var = 0;

/* This is used just to create some content that GDB can break on.  */
volatile int other_var = 0;

void
foo (void)
{
  while (global_var == 0)
    {
      sleep (1);
      other_var = 42;	/* Break here.  */
    }
}

int
main (void)
{
  gdb_watchdog (300);

  foo ();
  return 0;
}
