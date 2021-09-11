/* Copyright (C) 2021 Advanced Micro Devices, Inc. All rights reserved.

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

#include <cstdio>
#include <hip/hip_runtime.h>

__device__ void
base_case ()
{
  /* That printf is necessary to reproduce the exact failure as reported in
     SWDEV-294225.  */
  printf ("Hello device\n");
  return; /* break here */
}

template <unsigned int N>
__device__ void
deep ()
{
  deep<N-1> ();
}

template <>
__device__ void
deep<0> ()
{
  base_case ();
}

__global__ void
hip_deep ()
{
  deep<10> ();
}

int
main ()
{
  hipLaunchKernelGGL (HIP_KERNEL_NAME (hip_deep), dim3 (1), dim3 (1), 0, 0);
  hipDeviceSynchronize ();
  return EXIT_SUCCESS;
}

