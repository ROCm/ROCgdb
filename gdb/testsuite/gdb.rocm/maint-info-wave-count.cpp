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
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

#include <hip/hip_runtime.h>

/* A single workgroup, single workitem kernel occupies exactly one
   wave.  __forceinline__ and volatile asm guarantee the s_nop is
   emitted in-line, giving the debugger a reliable address to break
   on while the wave is still active.  */

__global__ void __attribute__ ((optnone))
single_wave_kernel ()
{
  __asm__ volatile ("s_nop 0" ::: );  /* Debugger breaks here.  */
}

int
main ()
{
  single_wave_kernel<<<1, 1>>> ();
  return hipDeviceSynchronize () != hipSuccess;
}
