/* This testcase is part of GDB, the GNU debugger.

   Copyright 2026 Free Software Foundation, Inc.

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
kernel (int *val1, int *val2)
{
  *val1 += 10;
  WAIT_MEM;
  *val2 += 100;
  WAIT_MEM;
  *val1 += 20;
  WAIT_MEM;
  *val2 += 200;
  WAIT_MEM;

  /* Some devices that don't support "precise memory" miss watchpoints when
     they would trigger near the end of the kernel.  Execute a bunch of sleeps
     to make sure this doesn't happen.  Just a handful of instructions should
     be enough, but this executes quickly anyway.  */
  for (int i = 0; i < 100000; ++i)
    __builtin_amdgcn_s_sleep (8);
}

/* Global pointers for the test to watch.  */
int *global_ptr1;
int *global_ptr2;
int host_global = 5;

int
main ()
{
  /* Break before runtime load.  */
  CHECK (hipMalloc (&global_ptr1, sizeof (int)));
  CHECK (hipMalloc (&global_ptr2, sizeof (int)));
  CHECK (hipMemset (global_ptr1, 0, sizeof (int)));
  CHECK (hipMemset (global_ptr2, 0, sizeof (int)));

  /* Break after malloc.  */
  kernel<<<1, 1>>> (global_ptr1, global_ptr2);
  CHECK (hipDeviceSynchronize ());

  host_global += 12;

  /* Break before second launch.  */
  kernel<<<1, 1>>> (global_ptr1, global_ptr2);
  CHECK (hipDeviceSynchronize ());

  CHECK (hipFree (global_ptr1));
  CHECK (hipFree (global_ptr2));
  return 0;
}
