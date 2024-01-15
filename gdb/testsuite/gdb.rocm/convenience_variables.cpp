/* Copyright (C) 2023-2024 Free Software Foundation, Inc.
   Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.

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

#include <stdio.h>
#include <hip/hip_runtime.h>

#define CHECK(cmd)                                                           \
  {                                                                          \
    hipError_t error = cmd;                                                  \
    if (error != hipSuccess)                                                 \
      {                                                                      \
	fprintf (stderr, "error: '%s'(%d) at %s:%d\n",                       \
		 hipGetErrorString (error), error,                           \
		 __FILE__, __LINE__);                                        \
	exit (EXIT_FAILURE);                                                 \
      }                                                                      \
  }

__global__ void
kern ()
{
  const int thread_idx_x = threadIdx.x;
  const int thread_idx_y = threadIdx.y;
  const int thread_idx_z = threadIdx.z;

  const int group_idx_x = blockIdx.x;
  const int group_idx_y = blockIdx.y;
  const int group_idx_z = blockIdx.z;

  const int group_size_x = blockDim.x;
  const int group_size_y = blockDim.y;
  const int group_size_z = blockDim.z;

  const int wave_num = (thread_idx_z * group_size_y * group_size_x
			+ thread_idx_y * group_size_x
			+ thread_idx_x);

  const int wave_in_group = wave_num / warpSize;
  const int lane_num = wave_num % warpSize;

  /* Break here.  */
}

int
main ()
{
  /* Each workgroup has 64 workitems, so exactly 1 or 2 waves per workgroup
     depending on the architecture.  */
  kern<<<dim3 (3, 4, 5), dim3 (8, 4, 2)>>> ();
  CHECK (hipDeviceSynchronize ());
}
