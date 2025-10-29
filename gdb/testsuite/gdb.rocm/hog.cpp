/* Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights reserved.

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

#define CHECK(cmd)                                                           \
    {                                                                        \
	hipError_t error = cmd;                                              \
	if (error != hipSuccess)                                             \
	  {                                                                  \
	    fprintf(stderr, "error: '%s'(%d) at %s:%d\n",                    \
		    hipGetErrorString(error), error,                         \
		    __FILE__, __LINE__);                                     \
	    exit(EXIT_FAILURE);                                              \
	  }                                                                  \
    }

__global__ void
hog_kernel ()
{
  while (true)
    __builtin_amdgcn_s_sleep (1);
}

int
main ()
{
  int deviceId;
  CHECK (hipGetDevice (&deviceId));

  hipDeviceProp_t props;
  CHECK (hipGetDeviceProperties (&props, deviceId));

  /* We use a single workgroup with as many workitems as the
     architecture allows which guarantees that when hog terminates
     resources are released allowing the hw to schedule new waves
     into vacated slots. */
  hog_kernel<<<1, props.maxThreadsPerBlock>>> ();
  CHECK (hipDeviceSynchronize ());
  return 0;
}

