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

/* '#pragma omp target data' wraps two consecutive '#pragma omp target'
   regions, sharing the device data buffer.  This is the natural way
   OpenMP applications stage data to the GPU and exercise multiple
   kernel launches against it.  */

#include <stdio.h>

#define N 32

static void
fill (int *p, int v)
{
  for (int i = 0; i < N; i++)
    p[i] = v + i;
}

int
main (void)
{
  int a[N];
  int b[N];
  int c[N];

  fill (a, 0);
  fill (b, 100);
  for (int i = 0; i < N; i++)
    c[i] = 0;

  #pragma omp target data map(to:a[0:N], b[0:N]) map(tofrom:c[0:N])
  {
    /* First kernel: c = a + b.  */
    #pragma omp target
    {
      for (int i = 0; i < N; i++)
	c[i] = a[i] + b[i];	/* data-k1 */
    }

    /* Second kernel: c *= 2 (data buffer reused, no host transfer).  */
    #pragma omp target
    {
      for (int i = 0; i < N; i++)
	c[i] = c[i] * 2;	/* data-k2 */
    }
  }

  printf ("c[0]=%d c[%d]=%d\n", c[0], N - 1, c[N - 1]);
  return 0;
}
