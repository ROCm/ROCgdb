/* Copyright (C) 2023-2024 Free Software Foundation, Inc.
   Copyright (C) 2023-2024 Advanced Micro Devices, Inc. All rights reserved.

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

#define CHECK(cmd)						\
  do								\
    {								\
      hipError_t error = cmd;					\
      if (error != hipSuccess)					\
	{							\
	  fprintf (stderr, "error: '%s'(%d) at %s:%d\n",	\
		   hipGetErrorString (error), error,		\
		   __FILE__, __LINE__);				\
	  exit (EXIT_FAILURE);					\
	}							\
    } while (0)

__global__ void
kern ()
{
  asm ("s_endpgm_insn: s_endpgm");
}

__global__ void
second_kernel ()
{
}

int
main ()
{
  /* Use 1-thread blocks to easily control number of waves.  */
  size_t blocksize = 1;
  size_t gridsize = 10;

  kern<<<gridsize, blocksize>>> ();

  /* Stopping at this second kernel after the first kernel completely
     finishes makes GDB refresh its thread list while the
     amd-dbgapi-target is still active, which triggers different code
     paths in GDB that lead to deleting exited threads.  We test both
     stopping here, and not stopping here.  */
  second_kernel<<<1, 1>>> ();

  CHECK (hipDeviceSynchronize ());

  return 0;
}
