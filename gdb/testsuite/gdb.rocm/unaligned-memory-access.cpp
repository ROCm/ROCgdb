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
    } while (0)

/* An array wrapped in a struct so we only need to specify the
   alignment once.  */
struct byte_array
{
  alignas (8) unsigned char array[8];
};

/* An array in global memory.  */
__device__ byte_array global = {
  0x11, 0x22, 0x33, 0x44,
  0x55, 0x66, 0x77, 0x88,
};

/* An array in LDS.  */
__shared__ byte_array shared;

__device__ void
done ()
{
}

__global__ void
kernel ()
{
  /* An array in the private_lane (swizzled) address space.  */
  byte_array local;

  /* __shared__ cannot have an initializer, so we initialize it
     explicitly here.  Do the same for the local array to keep things
     simple and consistent.  */
  memcpy (&shared, &global, sizeof (global));
  memcpy (&local, &global, sizeof (global));

  done (); /* break here */
}

int
main (int argc, char **argv)
{
  kernel<<<1, 1>>> ();
  CHECK (hipDeviceSynchronize ());
  return 0;
}
