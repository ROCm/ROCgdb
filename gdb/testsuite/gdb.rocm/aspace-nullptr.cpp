/* This testcase is part of GDB, the GNU debugger.

   Copyright (C) 2026 Free Software Foundation, Inc.
   Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

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
#include "gdb_watchdog.h"
#include "rocm-test-utils.h"

__global__ void
kern ()
{
  /* Define an LDS (i.e. local address space) variable to make sure
     the address space is allocated.  */
  __shared__ int local_var;
  if (threadIdx.x == 0)
    local_var = 42;

  NOP (1); /* Break here.  */
}

int
main()
{
  /* Make sure that if anything goes wrong, the program eventually
     gets killed.  */
  gdb_watchdog (30);

  /* Make dimensions large enough to create workgroups with multiple
     waves.  */
  kern<<<2, 320>>> ();
  CHECK (hipDeviceSynchronize ());

  return 0;
}
