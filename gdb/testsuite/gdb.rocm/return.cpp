/* Copyright (C) 2022 Free Software Foundation, Inc.
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

#include <cstdio>
#include <hip/hip_runtime.h>

#define CHECK(cmd)                                                           \
  {                                                                          \
    hipError_t error = cmd;                                                  \
    if (error != hipSuccess) {                                               \
	fprintf(stderr, "error: '%s'(%d) at %s:%d\n",                        \
		hipGetErrorString(error), error, __FILE__, __LINE__);        \
	  exit(EXIT_FAILURE);                                                \
    }                                                                        \
  }

template<typename T>
__device__ T
returnVal (T v)
{
  return v;
}

__device__ int
foo ()
{
  if (threadIdx.x == 2)
    {
      /* RETURN HERE.  */
      return 8;
    }
  /* RETURN2 HERE.  */
  return 6;
}

__device__ int
bar ()
{
  if (threadIdx.x % 2 == 0)
    return foo ();
  else
    return 0;
}

__device__ void
f ()
{
  if (threadIdx.x % 2 == 0)
    {
      int t1 = returnVal<int> (-1);
      short t2 = returnVal<short> (-2);
      unsigned short t3 = returnVal<unsigned short> (3);
      long t4 = returnVal<long> (-4);
      float t5 = returnVal<float> (5.0);
      long double t6 = returnVal<long double> (3.14);
      /* BREAK HERE.  */
      int brk = 0;
    }

  int returned_value = bar ();
  returned_value = bar ();
}

__global__ void
kernel ()
{
  f ();
}

int main ()
{
  hipLaunchKernelGGL(kernel, dim3(1), dim3(4), 0, 0);
  CHECK (hipDeviceSynchronize ());
}
