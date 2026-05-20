/* Copyright 2026 Free Software Foundation, Inc.
   Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

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

/* A small offloaded program with several distinguishable statements
   inside a '#pragma omp target' region, used to exercise single
   stepping on the GPU.  */

#include <stdio.h>

#define N 16

int
main (void)
{
  int a[N];
  int b[N];

  for (int i = 0; i < N; i++)
    {
      a[i] = i;
      b[i] = 0;
    }

  #pragma omp target map(to:a[0:N]) map(tofrom:b[0:N])
  {
    int x = 0;			/* step-1 */
    int y = 1;			/* step-2 */
    int z = a[0] + a[1];	/* step-3 */

    b[0] = x;
    b[1] = y;
    b[2] = z;			/* step-last */
  }

  printf ("b[0]=%d b[1]=%d b[2]=%d\n", b[0], b[1], b[2]);
  return 0;
}
