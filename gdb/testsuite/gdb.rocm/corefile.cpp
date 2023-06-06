/* Copyright (C) 2023 Free Software Foundation, Inc.
   Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.

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
#include <vector>

#define CHECK(cmd)                                                           \
  do                                                                         \
    {                                                                        \
      hipError_t error = cmd;                                                \
      if (error != hipSuccess)                                               \
	{                                                                    \
	  fprintf (stderr, "error: '%s'(%d) at %s:%d\n",                     \
		   hipGetErrorString (error), error, __FILE__, __LINE__);    \
	  exit (EXIT_FAILURE);                                               \
	}                                                                    \
    } while (0)

__device__ int
bar ()
{
  return threadIdx.x;
}

__device__ int
baz ()
{
  return 0;
}

__device__ int
foo ()
{
  if (blockIdx.x % 2 == 0)
    return bar ();
  else
    return baz ();
}

__global__ void
kern (int *ret, int n)
{
  size_t global_idx = threadIdx.x + (blockIdx.x * blockDim.x);
  if (global_idx < n)
    ret[global_idx] = foo ();
}

/* Secondary kernel, meant to run concurrently on a separate stream.  */

__global__ void
aux_kernel ()
{
  while (1)
    __builtin_amdgcn_s_sleep (1);
}

constexpr int num_waves = 2;

int
main (int argc, char* argv[])
{
  int device_id;
  CHECK (hipGetDevice (&device_id));
  hipDeviceProp_t props;
  CHECK (hipGetDeviceProperties (&props, device_id));

  int *dev_buf;
  CHECK (hipMalloc (&dev_buf, num_waves * props.warpSize * sizeof (int)));

  hipStream_t st1;
  hipStream_t st2;
  CHECK (hipStreamCreate (&st1));
  CHECK (hipStreamCreate (&st2));

  aux_kernel<<<num_waves, props.warpSize, 0, st1>>> ();
  kern<<<num_waves, props.warpSize, 0, st2>>> (dev_buf,
					       num_waves * props.warpSize);
  CHECK (hipStreamSynchronize (st2));

  std::vector<int> result (num_waves * props.warpSize);
  CHECK (hipMemcpy (result.data (), dev_buf,
		    num_waves * props.warpSize * sizeof (int),
		    hipMemcpyDeviceToHost));

}
