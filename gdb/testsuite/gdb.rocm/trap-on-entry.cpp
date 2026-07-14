/* This testcase is part of GDB, the GNU debugger.

   Copyright 2022-2026 Free Software Foundation, Inc.

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

#include "hip/hip_runtime.h"
#include <cassert>

__global__ void
do_an_addition (int a, int b, int *out)
{
  *out = a + b;
}

static void
run_addition (int *result_ptr)
{
  do_an_addition<<<dim3(1), dim3(1), 0, 0>>> (1, 2, result_ptr);

  hipError_t error = hipDeviceSynchronize ();
  assert (error == hipSuccess);
}

static void
enable_trap_on_entry ()
{
}

int
main ()
{
  int *result_ptr;

  hipError_t error = hipMalloc (&result_ptr, sizeof (int));
  assert (error == hipSuccess);

  /* Dry run: this first dispatch triggers internal kernels (heap
     setup, etc.).  */
  run_addition (result_ptr);

  /* The debugger enables trap-on-entry here.  */
  enable_trap_on_entry ();

  /* This dispatch is expected to trap on entry.  */
  run_addition (result_ptr);

  /* Re-run the kernel after disabling trap-on-entry.  */
  run_addition (result_ptr);

  return 0;
}
