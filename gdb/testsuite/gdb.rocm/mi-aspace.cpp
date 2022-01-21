/* Copyright 2022 Free Software Foundation, Inc.
   Copyright (C) 2022 Advanced Micro Devices, Inc. All rights reserved.

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

__device__ void
aspace_test (unsigned gid)
{
}

__global__ void
kernel ()
{
  unsigned gid0 = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
  aspace_test (gid0);
}

int
main ()
{
  dim3 grid_dim (1);
  dim3 block_dim (32);

  hipLaunchKernelGGL (kernel, grid_dim, block_dim, 0, 0);

  hipDeviceSynchronize ();

  return 0;
}
