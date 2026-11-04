/* Copyright 2023-2026 Free Software Foundation, Inc.
   Copyright (C) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.

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
#include <cassert>
#include <stdio.h>
#include <stdlib.h>

#include "rocm-test-utils.h"

/* This is initialized by the kernel function below.  */
__device__ size_t *global_ptr = nullptr;

/* This is initialized by GDB.  */
__device__ size_t *global_ptr2 = nullptr;

__device__ void
stop ()
{
}

__global__ void
kernel ()
{
  size_t local_var = blockIdx.x * blockDim.x + threadIdx.x;

  /* This will be zero for the first HIP thread, or lane.  */
  if (local_var == 0)
    global_ptr = &local_var;

  /* Just make the var non-zero.  */
  local_var++;

  /* Ensure thread 0 has assigned to global_ptr before the other
     thread attempts to read it.  Even though the lanes are in the
     same wave, this prevents the compiler from restructuring the
     control flow graph in a way that would break the test.  */
  __syncthreads ();

  stop (); /* break-here */
}

int
main ()
{
  /* We only need more than one lane.  Two is sufficient.  */
  kernel<<<dim3 (1), dim3 (2)>>> ();
  CHECK (hipDeviceSynchronize ());
  return 0;
}
