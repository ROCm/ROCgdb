/* This testcase is part of GDB, the GNU debugger.

   Copyright 2025 Free Software Foundation, Inc.

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

struct S
{
  int x;
  int y;
  int z;
};

__device__ S my_s;

__device__ void
foo (int arg1, int arg2, int x)
{
}

__device__ void
bar (int arg1, int arg2, int x)
{
}

__global__ void
my_kernel ()
{
  while (1)
    {
      if (threadIdx.x & 1)
	foo (1, 2, threadIdx.x);
      else
	bar (3, 4, threadIdx.x);
    }
}

int
main ()
{
  my_kernel<<<dim3 (4), dim3 (4), 0, 0>>> ();
  if (hipDeviceSynchronize () != hipSuccess)
    return 1;
  return 0;
}
