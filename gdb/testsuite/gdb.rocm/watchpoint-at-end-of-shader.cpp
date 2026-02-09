/* This testcase is part of GDB, the GNU debugger.

   Copyright 2024-2026 Free Software Foundation, Inc.

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
set_val (int *p, int v)
{
  *p = v;
}

int
main ()
{
  int *v;
  CHECK (hipMalloc (&v, sizeof (*v)));

  /* First dispatch to initialize the memory.  */
  set_val<<<1, 1>>> (v, 64);
  CHECK (hipDeviceSynchronize ());

  /* Break here.  */
  set_val<<<1, 1>>> (v, 8);

  CHECK (hipDeviceSynchronize ());

  return 0;
}
