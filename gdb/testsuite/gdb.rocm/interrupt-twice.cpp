/* Copyright 2026 Free Software Foundation, Inc.

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

#include <hip/hip_runtime.h>
#include "gdb_watchdog.h"

__device__ void
loop ()
{
  while (true)
    __builtin_amdgcn_s_sleep (8);
}

__global__ void
kern ()
{
  loop ();
}

int
main ()
{
  /* Make sure that if anything goes wrong, the program eventually
     gets killed.  */
  gdb_watchdog (30);

  kern<<<1, 1>>> ();
  return hipDeviceSynchronize () != hipSuccess;
}
