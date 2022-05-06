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

#include <stdio.h>
#include <hip/hip_runtime.h>

__device__ int extern_global;
__device__ static int static_global;

__device__ static void
done ()
{
}

__device__ static void
set_globals ()
{
  extern_global = 1;
  static_global = 2;

  printf ("extern_global's address on device: %p\n", &extern_global);
  printf ("static_global's address on device: %p\n", &static_global);

  done ();
}

__global__ void
kernel ()
{
  set_globals ();
}

int
main (int argc, char* argv[])
{
  /* Reference the static global's address from the host side to force
     externalization.  Making "set_globals" __device__ __host__ would
     also do it at the time of writing, and it was how the issue was
     originally discovered, but taking the address seems more robust,
     as externalization is necessary mainly because
     hipGetSymbolAddress exists.  */
  printf ("extern_global's address on host: %p\n", &extern_global);
  printf ("static_global's address on host: %p\n", &static_global);

  hipLaunchKernelGGL (kernel, dim3 (1), dim3 (1), 0, 0);
  hipDeviceSynchronize ();
  return 0;
}
