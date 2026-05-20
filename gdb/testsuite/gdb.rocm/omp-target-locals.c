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

/* Exercise inspecting variables in different OpenMP data-sharing
   classes (mapped, private, firstprivate) from inside a
   '#pragma omp target' region offloaded to the GPU.  */

#include <stdio.h>

#define N 8

int
main (void)
{
  int in_arr[N];
  int out_arr[N];

  for (int i = 0; i < N; i++)
    {
      in_arr[i] = 100 + i;
      out_arr[i] = 0;
    }

  int firstpriv_val = 42;

  #pragma omp target						\
	      map(to:in_arr[0:N]) map(tofrom:out_arr[0:N])	\
	      firstprivate(firstpriv_val)
  {
    int priv_val = 7;		/* locals-line */

    for (int i = 0; i < N; i++)
      out_arr[i] = in_arr[i] + priv_val + firstpriv_val;
  }

  printf ("out[0]=%d out[%d]=%d\n", out_arr[0], N - 1, out_arr[N - 1]);
  return 0;
}
