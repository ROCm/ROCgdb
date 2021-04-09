/* Copyright 2021 Free Software Foundation, Inc.
   Copyright (C) 2015-2021 Advanced Micro Devices, Inc. All rights reserved.

   This test program is part of GDB, the GNU debugger.

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

#include <limits.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>

int main (int argc, char ** argv)
{
  char prog[PATH_MAX];
  int len;

  strcpy (prog, argv[0]);
  len = strlen (prog);
  /* Replace "exec-bit-extract" with "bit-extract".  */
  memcpy (prog + len - 16, "bit-extract", 11);
  prog[len - 5] = 0;

  /* exec bit-extract  */
  execl (prog, prog, (char *) NULL);
  exit (1);
}
