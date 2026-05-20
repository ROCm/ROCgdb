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

/* Minimal OpenMP "target" offload program used to exercise
   breakpoints in a device kernel offloaded by an OpenMP application.  */

#include <stdio.h>
#include <stdlib.h>

#define N 64

int
main (void)
{
  int a[N];
  int b[N];
  int c[N];

  for (int i = 0; i < N; i++)
    {
      a[i] = i;
      b[i] = 2 * i;
      c[i] = 0;
    }

  /* The map clauses move A and B onto the device, and C tofrom the
     device.  The body of the target region is executed on the GPU.  */
  #pragma omp target map(to:a[0:N], b[0:N]) map(tofrom:c[0:N])
  {
    for (int i = 0; i < N; i++)
      c[i] = a[i] + b[i];	/* break-target-line */
  }

  int sum = 0;
  for (int i = 0; i < N; i++)
    sum += c[i];

  printf ("sum=%d\n", sum);
  return sum == (3 * (N - 1) * N / 2) ? 0 : 1;
}
