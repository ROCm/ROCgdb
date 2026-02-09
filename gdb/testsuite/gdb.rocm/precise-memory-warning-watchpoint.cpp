/* This testcase is part of GDB, the GNU debugger.

   Copyright 2021-2026 Free Software Foundation, Inc.

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

#include "rocm-test-utils.h"

__global__ void
kernel (int *ptr)
{
  for (int i = 0; i < 1000; ++i)
    {
      (*ptr)++;
    }
}

int *global_ptr;

int
main (int argc, char* argv[])
{
  CHECK (hipMalloc (&global_ptr, sizeof (int)));
  CHECK (hipMemset (global_ptr, 0, sizeof (int)));

  /* Break here.  */
  hipLaunchKernelGGL (kernel, dim3 (1), dim3 (1), 0, 0, global_ptr);
  CHECK (hipDeviceSynchronize ());

  CHECK (hipFree (global_ptr));
  return 0;
}
