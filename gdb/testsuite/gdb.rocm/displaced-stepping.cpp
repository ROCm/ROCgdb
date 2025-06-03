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

      "END_OF_COMMON_TESTS:					  \n\t"
      "	  s_nop		1" ::: "s4", "s5", "s6", "s7");

  asm(".if !(.amdgcn.gfx_generation_number < 12 ||    \
	     (.amdgcn.gfx_generation_number == 12 &&  \
	      .amdgcn.gfx_generation_minor < 5))		  \n\t"

      "	  .globl ADD_PC_PLUS_POS_IMM				  \n\t"
      "	  .globl ADD_PC_PLUS_NEG_IMM				  \n\t"
      "	  .globl ADD_PC_PLUS_POS_REGVAL				  \n\t"
      "	  .globl ADD_PC_PLUS_NEG_REGVAL				  \n\t"

      "ADD_PC_TST1:						  \n\t"
      "# test s8=1: pc+imm					  \n\t"
      "	  s_mov_b32	s8, 1					  \n\t"
      "ADD_PC_PLUS_POS_IMM:					  \n\t"
      "	  s_add_pc_i64	ADD_PC_TST2 - ADD_PC_TST1_END		  \n\t"
      "ADD_PC_TST1_END:						  \n\t"
      "	  s_trap	2					  \n\t"

      "ADD_PC_TST4:						  \n\t"
      "# test s8=4: pc+reg					  \n\t"
      "	  s_add_i32	s8, s8, 1				  \n\t"
      "	  s_mov_b64	s[6:7], ADD_PC_TST_END - ADD_PC_TST4_END  \n\t"
      "ADD_PC_PLUS_POS_REGVAL:					  \n\t"
      "	  s_add_pc_i64	s[6:7]					  \n\t"
      "ADD_PC_TST4_END:						  \n\t"
      "	  s_trap	2					  \n\t"

      "ADD_PC_TST3:						  \n\t"
      "# test s8=3: pc-reg					  \n\t"
      "	  s_add_i32	s8, s8, 1				  \n\t"
      "	  s_mov_b64	s[4:5], ADD_PC_TST4 - ADD_PC_TST3_END	  \n\t"
      "ADD_PC_PLUS_NEG_REGVAL:					  \n\t"
      "	  s_add_pc_i64	s[4:5]					  \n\t"
      "ADD_PC_TST3_END:						  \n\t"
      "	  s_trap	2					  \n\t"

      "ADD_PC_TST2:						  \n\t"
      "# test s8=2: pc-imm					  \n\t"
      "	  s_add_i32	s8, s8, 1				  \n\t"
      "ADD_PC_PLUS_NEG_IMM:					  \n\t"
      "	  s_add_pc_i64	ADD_PC_TST3 - ADD_PC_TST2_END		  \n\t"
      "ADD_PC_TST2_END:						  \n\t"
      "	  s_trap	2					  \n\t"

      "ADD_PC_TST_END:						  \n\t"
      "	  s_nop  2						  \n\t"
      ".endif" ::: "s4", "s5", "s6", "s7", "s8");
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
