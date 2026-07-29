/* Copyright (C) 2026 Free Software Foundation, Inc.
   Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

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

#include <hip/hip_runtime.h>

/* Regular inline function - inline frame should be skipped during stepping.  */
__device__ __attribute__((always_inline)) inline int
add_numbers (int a, int b)
{
  return a + b;
}

__global__ void
test_trap_kernel ()
{
  int x = 1;  /* Breakpoint here.  */
  int y = 2;
  int z = add_numbers (x, y);  /* Step over inline function.  */
  __builtin_verbose_trap ("check verbose", "This is verbose trap!");
}

int
main ()
{
  test_trap_kernel<<<1, 1>>> ();
  return hipDeviceSynchronize () != hipSuccess;
}
