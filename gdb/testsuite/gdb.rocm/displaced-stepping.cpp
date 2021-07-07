/* Copyright (C) 2020-2021 Free Software Foundation, Inc.
   Copyright (C) 2020-2021 Advanced Micro Devices, Inc. All rights reserved.

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

__global__ void
kernel ()
{
  asm("s_nop	0");

  asm("s_getpc_b64	[s0, s1]\n\t" /* getpc breakpoint here */
      "s_add_u32	s2, s0, 16\n\t"
      "s_addc_u32	s3, s1, 0");

  asm("s_swappc_b64	[s0, s1], [s2, s3]\n\t" /* swappc breakpoint here */
      "s_trap	2");

  asm("s_call_b64	[s0,s1], 1\n\t" /* call breakpoint here */
      "s_trap	2\n\t"
      "s_add_u32	s2, s0, 20\n\t"
      "s_addc_u32	s3, s1, 0");

  asm("s_setpc_b64	[s2, s3]\n\t" /* setpc breakpoint here */
      "s_trap	2");

  asm("s_nop	0"); /* last breakpoint here */
}

int
main (int argc, char **argv)
{
  hipLaunchKernelGGL (kernel, dim3 (1), dim3 (1), 0 /*dynamicShared*/,
		      0 /*stream*/);

  /* Wait until kernel finishes.  */
  hipDeviceSynchronize ();

  return 0;
}
