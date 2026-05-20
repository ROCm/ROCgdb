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

/* A single binary that contains two distinct '#pragma omp declare
   target' device functions and launches them from two separate
   '#pragma omp target' regions.  Used to verify that GDB can place
   breakpoints in named device functions and distinguish between
   them.  */

#include <stdio.h>

#define N 32

#pragma omp declare target
static int
square_dev (int x)
{
  return x * x;		/* sq-line */
}

static int
cube_dev (int x)
{
  return x * x * x;	/* cb-line */
}
#pragma omp end declare target

int
main (void)
{
  int in[N];
  int out_sq[N];
  int out_cb[N];

  for (int i = 0; i < N; i++)
    {
      in[i] = i + 1;
      out_sq[i] = 0;
      out_cb[i] = 0;
    }

  #pragma omp target map(to:in[0:N]) map(tofrom:out_sq[0:N])
  for (int i = 0; i < N; i++)
    out_sq[i] = square_dev (in[i]);

  #pragma omp target map(to:in[0:N]) map(tofrom:out_cb[0:N])
  for (int i = 0; i < N; i++)
    out_cb[i] = cube_dev (in[i]);

  printf ("sq[0]=%d cb[0]=%d sq[%d]=%d cb[%d]=%d\n",
	  out_sq[0], out_cb[0],
	  N - 1, out_sq[N - 1],
	  N - 1, out_cb[N - 1]);
  return 0;
}
