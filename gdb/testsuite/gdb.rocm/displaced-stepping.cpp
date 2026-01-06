/* Copyright 2022-2026 Free Software Foundation, Inc.
   Copyright (C) 2020-2026 Advanced Micro Devices, Inc. All rights reserved.

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

__global__ void
kernel ()
{
  asm("	  .globl GETPC			    			  \n\t"
      "	  .globl SWAPPC	      					  \n\t"
      "	  .globl CALL	     		    			  \n\t"
      "	  .globl SETPC	      		    			  \n\t"
      "	  .globl END_OF_TESTS					  \n\t"

      "	  s_nop		0					  \n\t"

      "GETPC:	    						  \n\t"
      "	  s_getpc_b64	[s4, s5]				  \n\t"
      "	  s_add_u32	s6, s4, 16				  \n\t"
      "	  s_addc_u32	s7, s5, 0				  \n\t"

      "SWAPPC:	      						  \n\t"
      "	  s_swappc_b64	[s4, s5], [s6, s7]			  \n\t"
      "	  s_trap	2					  \n\t"

      "CALL:	    						  \n\t"
      "	  s_call_b64	[s4, s5], 1				  \n\t"
      "	  s_trap		2				  \n\t"
      "	  s_add_u32	s6, s4, 20				  \n\t"
      "	  s_addc_u32	s7, s5, 0				  \n\t"

      "SETPC:	    						  \n\t"
      "	  s_setpc_b64	[s6, s7]				  \n\t"
      "	  s_trap	2					  \n\t"

      "END_OF_TESTS:						  \n\t"
      "	  s_nop		1" ::: "s4", "s5", "s6", "s7");
}

int
main (int argc, char **argv)
{
  hipLaunchKernelGGL (kernel, dim3 (1), dim3 (1), 0 /*dynamicShared*/,
		      0 /*stream*/);

  /* Wait until kernel finishes.  */
  CHECK (hipDeviceSynchronize ());

  return 0;
}
