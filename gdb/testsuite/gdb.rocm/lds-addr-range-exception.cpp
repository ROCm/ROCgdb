/* Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.

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
#include <stdio.h>
#include <stdlib.h>
#include <cassert>
#include "rocm-test-utils.h"

__global__ void
kern (size_t dyn_alloc, size_t lds_size)
{
  extern __shared__ char arr[];

  /* Access up to dyn_alloc should always work.  */
  for (size_t i = 0; i < dyn_alloc; i++)
    arr[i] = i;

  __syncthreads ();

  for (size_t i = 0; i < dyn_alloc; i++)
    assert (arr[i] == i);

  /* Use the last four bytes of the LDS  */
  size_t idx = lds_size - 4;
  if (threadIdx.x == 0)
    {
      assert (idx >= dyn_alloc);
      arr[idx] = 8;
    }
  __syncthreads ();

  /* This is expected to fail.
     One could run this kernel once and expect to fail at the assert, then run
     it again with the LDS reporting on, and check you receive the memviol.  */
  if (threadIdx.x == 0)
    assert (arr[idx] == 8);
  __syncthreads ();
}

int
main (int argc, char* argv[])
{
  hipDeviceProp_t props;
  int deviceId;
  CHECK (hipGetDevice (&deviceId));
  CHECK (hipGetDeviceProperties(&props, deviceId));

  constexpr size_t shared_mem_alloc = 64;
  size_t lds_size = props.sharedMemPerBlock;
  kern<<<1, 128, shared_mem_alloc, 0>>> (shared_mem_alloc, lds_size);
  CHECK (hipDeviceSynchronize ());
  return 0;
}