/* Copyright (C) 2022 Advanced Micro Devices, Inc. All rights reserved.

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

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

static void
break_here (void)
{}

int
main (void)
{
  /* Make sure we don't run forever.  */
  alarm (30);

  for (int i = 0; i < 100; ++i)
    break_here ();

  /* Write that file so the test knows this program has successfully
     ran to completion after the detach.  */
  FILE *f = fopen (TOUCH_FILE_PATH, "w");
  if (f == NULL)
    abort ();

  fclose (f);

  return 0;
}
