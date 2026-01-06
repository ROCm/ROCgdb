/* Copyright (C) 2022-2026 Free Software Foundation, Inc.
   Copyright (C) 2022-2026 Advanced Micro Devices, Inc. All rights reserved.

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

   Test writes to default address space (global memory) as well as
   private_lane address space to test the watchpoints in different
   address spaces.

   The WAIT_MEM macro is used to force out a watchpoint hit event being
   reported by the target on specific synchronisation points for a more
   controlled execution.  */

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK(cmd)                                                           \
  {                                                                          \
    hipError_t error = cmd;                                                  \
    if (error != hipSuccess)                                                 \
      {                                                                      \
	fprintf (stderr, "error: '%s'(%d) at %s:%d\n",                       \
		 hipGetErrorString (error), error, __FILE__, __LINE__);      \
	exit (EXIT_FAILURE);                                                 \
      }                                                                      \
  }

#define WAIT_MEM						\
  asm volatile (".if .amdgcn.gfx_generation_number < 10\n"	\
		"  s_waitcnt 0\n"				\
		".elseif .amdgcn.gfx_generation_number < 11\n"	\
		"  s_waitcnt_vscnt null, 0\n"			\
		".else\n"					\
		"  s_wait_idle\n"				\
		".endif")

__device__ void
change_memory (char *private_ptr, int *global_ptr)
{
  /* Test if an address context (thread and SIMD lane) is handled
     correctly depending on the address space that that address
     belongs to.  */
  *(private_ptr + 3) = 'c';
  WAIT_MEM;

  *private_ptr = 'c';
  WAIT_MEM;

  *global_ptr = 2;
  WAIT_MEM;
}

/* Kernel entry point.  Initialise and change the addresses in different
   address spaces to trigger hardware watchpoints.  */
__global__ void
kernel (int *global_ptr)
{
  /* Initialise an array in the private_lane address space.  */
  char array[4] = {'a', 'a', 'a', 'a'};

  /* Only testing watchpoints in global and private_lane address
     spaces.  */
  array[3] = 'b';
  WAIT_MEM;

  array[0] = 'b';
  WAIT_MEM;

  *global_ptr = 1;
  WAIT_MEM;

  change_memory (&array[0], global_ptr);

  array[3] = 'd';
  WAIT_MEM;

  *global_ptr = 3;
  WAIT_MEM;
}

int
main (int argc, char* argv[])
{
  int *global_ptr;
  CHECK (hipMalloc (&global_ptr, 4));
  int init_value_h = 0;
  CHECK (hipMemcpy (global_ptr, &init_value_h, sizeof (init_value_h),
		    hipMemcpyHostToDevice));

  kernel<<<2, 1, 0>>> (global_ptr);
  CHECK (hipDeviceSynchronize ());
  return 0;
}
