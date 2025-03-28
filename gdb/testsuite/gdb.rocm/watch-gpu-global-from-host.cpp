/* This testcase is part of GDB, the GNU debugger.

   Copyright 2025-2026 Free Software Foundation, Inc.

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

__device__ int global = 0;

__global__ void
kern ()
{
}

int
main (int argc, char* argv[])
{
  kern<<<1, 1>>> ();
  if (hipDeviceSynchronize () != hipSuccess)
    return 1;

  int *devGlobal;
  if (hipGetSymbolAddress (reinterpret_cast<void **> (&devGlobal), global))
    return 2;

  /* Now update the device global from a CPU thread.  */
  *devGlobal = 8;
  return 0;
}
