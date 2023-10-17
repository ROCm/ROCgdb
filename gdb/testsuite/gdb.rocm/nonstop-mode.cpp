/* Copyright (C) 2019-2023 Free Software Foundation, Inc.
   Copyright (C) 2019-2023 Advanced Micro Devices, Inc. All rights reserved.

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

#include <cstdio>
#include <hip/hip_runtime.h>

#if !defined(GRID_DIM)
# error "Missing definition of GRID_DIM"
#endif
#if !defined(BLOCK_DIM)
# error "Missing definition of BLOCK_DIM"
#endif

#define CHECK(cmd)                                                           \
  {                                                                          \
    hipError_t error = cmd;                                                  \
    if (error != hipSuccess)                                                 \
      {                                                                      \
	fprintf (stderr, "error: '%s'(%d) at %s:%d\n",                       \
		 hipGetErrorString (error), error, __FILE__, __LINE__);      \
	exit (EXIT_FAILURE);                                                 \
      }                                                                      \
  }

/* Number of elements in Array.  */
constexpr size_t N = 64;

__global__ void
VectorAdd (int *d_a, int *d_b, int *d_c)
{
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < N)
    d_c[tid] = d_a[tid] + d_b[tid];
}

int
main ()
{
  int gpu_id;
  CHECK (hipGetDevice (&gpu_id));

  hipDeviceProp_t prop;
  CHECK (hipGetDeviceProperties (&prop, gpu_id));
  printf ("#   device %d [%04x:%02x.%02x] %s\n", gpu_id, prop.pciDomainID,
	  prop.pciBusID, prop.pciDeviceID, prop.name);

  /* Host allocations.  */
  int h_a[N] = {};
  int h_b[N] = {};
  int h_c[N] = {};
  for (int i = 0; i < N; ++i)
    {
      h_a[i] = 2 * i;
      h_b[i] = i;
    }

  /* Device allocations.  */
  int *d_a = nullptr;
  int *d_b = nullptr;
  int *d_c = nullptr;

  CHECK (hipMalloc (&d_a, sizeof (int) * N));
  CHECK (hipMalloc (&d_b, sizeof (int) * N));
  CHECK (hipMalloc (&d_c, sizeof (int) * N));

  CHECK (hipMemcpy (d_a, h_a, sizeof (int) * N, hipMemcpyHostToDevice));
  CHECK (hipMemcpy (d_b, h_b, sizeof (int) * N, hipMemcpyHostToDevice));

  VectorAdd<<<dim3 (GRID_DIM), dim3 (BLOCK_DIM)>>> (d_a, d_b, d_c);

  CHECK (hipMemcpy (h_c, d_c, sizeof (int) * N, hipMemcpyDeviceToHost));

  CHECK (hipFree (d_a));
  CHECK (hipFree (d_b));
  CHECK (hipFree (d_c));

  bool error_found = false;
  for (int i = 0; i < N; ++i)
    {
      if (h_a[i] + h_b[i] != h_c[i])
	{
	  fprintf (stderr, "%d + %d != %d (at index %d)", h_a[i], h_b[i],
		   h_c[i], i);
	  error_found = true;
	}
    }

  return (error_found ? EXIT_FAILURE : EXIT_SUCCESS);
}
