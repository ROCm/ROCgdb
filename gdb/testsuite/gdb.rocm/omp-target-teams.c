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

/* Use 'target teams distribute parallel for' so that GDB sees
   multiple GPU waves at the same breakpoint.  */

#include <stdio.h>

#define N 256

int
main (void)
{
  int a[N];
  int b[N];
  int c[N];

  for (int i = 0; i < N; i++)
    {
      a[i] = i;
      b[i] = N - i;
      c[i] = 0;
    }

  #pragma omp target teams distribute parallel for	\
	      map(to:a[0:N], b[0:N]) map(tofrom:c[0:N])
  for (int i = 0; i < N; i++)
    {
      c[i] = a[i] + b[i];	/* teams-line */
    }

  int correct = 1;
  for (int i = 0; i < N; i++)
    {
      if (c[i] != N)
	correct = 0;
    }

  printf ("correct=%d\n", correct);
  return correct ? 0 : 1;
}
