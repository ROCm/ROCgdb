/* This testcase is part of GDB, the GNU debugger.

   Copyright 2021-2026 Free Software Foundation, Inc.

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

#ifndef ROCM_TEST_UTILS_H
#define ROCM_TEST_UTILS_H

#include <stdio.h>
#include <stdlib.h>

/* Check the return value of a HIP call, exit on error.  */

#define CHECK(cmd)							\
  do									\
    {									\
      hipError_t error = cmd;						\
      if (error != hipSuccess)						\
	{								\
	  fprintf (stderr, "error: '%s'(%d) at %s:%d\n",		\
		   hipGetErrorString (error), error, __FILE__,		\
		  __LINE__);						\
	  exit (EXIT_FAILURE);						\
	}								\
    }									\
  while (0)

/* Ensure that all memory operations are completed before continuing,
   even when "precise-memory" is off.  */

#define WAIT_MEM							\
  asm volatile (".if .amdgcn.gfx_generation_number < 10\n"		\
		"  s_waitcnt 0\n"					\
		".elseif .amdgcn.gfx_generation_number < 11\n"		\
		"  s_waitcnt_vscnt null, 0\n"				\
		".else\n"						\
		"  s_wait_idle\n"					\
		".endif")

#endif /* ROCM_TEST_UTILS_H */
