/* Copyright (C) 2022-2023 Advanced Micro Devices, Inc. All rights reserved.

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

__device__ void
foo ()
{
}

__device__ static bool lock = true;

__global__ void
kern ()
{
  while (__hip_atomic_load (&lock, __ATOMIC_RELAXED,
			    __HIP_MEMORY_SCOPE_AGENT))
    {
      if (blockIdx.x == 0)
	{
	  foo ();
	  __hip_atomic_store (&lock, false, __ATOMIC_RELAXED,
			      __HIP_MEMORY_SCOPE_AGENT);
	}
      __builtin_amdgcn_s_sleep (1);
    }
}

int
main (int argc, char *argv[])
{
  kern<<<2, 1>>> ();
  CHECK (hipDeviceSynchronize ());
  return 0;
}
