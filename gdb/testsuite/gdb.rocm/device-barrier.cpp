/* This testcase is part of GDB, the GNU debugger.

   Copyright 2023-2024 Free Software Foundation, Inc.

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

#include "hip/hip_runtime.h"
#include <cstdio>
#include <vector>

#define CHECK(cmd)                                                           \
  do {                                                                       \
    hipError_t error = cmd;                                                  \
    if (error != hipSuccess)                                                 \
      {                                                                      \
	fprintf (stderr, "error: '%s'(%d) at %s:%d\n",                       \
		 hipGetErrorString (error), error, __FILE__, __LINE__);      \
	exit (EXIT_FAILURE);                                                 \
      }                                                                      \
  } while (0)

__global__ void
kern ()
{
  const size_t id = threadIdx.x;
  if (id == 0)
    {
      /* Introduce some delay for the first wave so other waves can reach
	 the barrier while the first wave is still in this block.  */
      for (int i = 0; i < 100; i++)
	__builtin_amdgcn_s_sleep (8);
      /* Break here.  */
    }
  __syncthreads ();
}

/* Use 128 threads to be sure to have at least 2 waves on all
   architectures.  */
constexpr size_t group_size = 128;

int
main ()
{
  kern<<<1, group_size>>> ();
  CHECK(hipDeviceSynchronize ());

  return EXIT_SUCCESS;
}
