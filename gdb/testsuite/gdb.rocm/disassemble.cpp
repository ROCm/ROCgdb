/* Copyright (C) 2021-2024 Free Software Foundation, Inc.
   Copyright (C) 2021-2024 Advanced Micro Devices, Inc. All rights reserved.

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
   along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <hip/hip_runtime.h>

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

/* Kernel entry point. */
__global__ void kernel ()
{
  int tid = threadIdx.x;
  tid += 1;
  tid += 1;
}

int
main (int argc, char* argv[])
{
  hipLaunchKernelGGL (kernel, dim3 (1), dim3 (1),
      0 /*dynamicShared*/, 0 /*stream*/);

  CHECK (hipDeviceSynchronize ());

  return 0;
}
