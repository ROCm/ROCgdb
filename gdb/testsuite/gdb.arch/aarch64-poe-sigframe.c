/* This test program is part of GDB, the GNU debugger.

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

/* Exercise AArch64's POE Extension signal frame unwinding.  */

/* This test is based on the Linux kernel documentation for Memory
   Protection Keys, including the arm64 Permission Overlay Extension
   (FEAT_S1POE) support described in
   Documentation/core-api/protection-keys.rst.  */

#include <sys/auxv.h>
#include <stdlib.h>
#include <signal.h>

static int count = 0;

static void
handler (int sig)
{
  count++;
}

int
main (int argc, char **argv)
{
  signal (SIGUSR1, handler);

  raise (SIGUSR1);

  return 0;
}
