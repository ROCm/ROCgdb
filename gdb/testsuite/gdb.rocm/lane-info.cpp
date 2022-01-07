/* Copyright 2021-2022 Free Software Foundation, Inc.
   Copyright (C) 2021-2022 Advanced Micro Devices, Inc. All rights reserved.

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
#include <unistd.h>

/* The kernel never returns, via this sleep, so that the .exp file can
   test background execution (cont&).  */

__device__ static void
sleep_forever ()
{
  while (1)
    __builtin_amdgcn_s_sleep (1);
}

__device__ static void
foo ()
{
}

__device__ static void
bar ()
{
  sleep_forever ();
}

__global__ void
kernel ()
{
  int tid_x = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

  if (tid_x % 2)
    foo ();
  else
    bar ();
}

int
main ()
{
  alarm (30);

 /* If the wavefront size is 64 lanes, then this results in 2 waves, 1
     with 64 lanes used, and 1 with 5 lanes used.  If the wavefront
     size is 32 lanes, then this results in 3 waves, 2 with 32 lanes
     used each, and 1 with 5 lanes used.  */
  hipLaunchKernelGGL (kernel, dim3 (1), dim3 (64 + 5),
		      0 /*dynamicShared*/, 0 /*stream*/);

  hipDeviceSynchronize ();

  return 0;
}
