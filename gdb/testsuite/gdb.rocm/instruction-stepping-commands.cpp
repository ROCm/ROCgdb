/* Copyright (C) 2024-2026 Free Software Foundation, Inc.
   Copyright (C) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.

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
    }								\
  while (0)

/* "noinline" so the nexti test can step into a real function call;
   "optnone" so the body has stable instructions for stepi/nexti and
   the locally-initialized value is not folded into the return.  */
__device__ static __attribute__ ((noinline, optnone))
int
return_zero ()
{
  int ret = 0;
  return ret;
}

/* "optnone" so each statement below maps to its own instruction
   sequence and the "var += 1" lines are not folded together --
   stepi/nexti rely on stepping through them one at a time.  */
__global__ void __attribute__ ((optnone))
kernel ()
{
  int var = return_zero ();
  var += 1; /* next line */
  var += 1;
}

int
main (int argc, char* argv[])
{
  kernel<<<1, 1>>> ();
  CHECK (hipDeviceSynchronize ());
  return 0;
}
