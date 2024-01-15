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
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

#include <hip/hip_runtime.h>
#include <vector>

#define CHECK(cmd)                                                           \
  do                                                                         \
    {                                                                        \
      hipError_t error = cmd;                                                \
      if (error != hipSuccess)                                               \
	{                                                                    \
	  fprintf (stderr, "error: '%s'(%d) at %s:%d\n",                     \
		   hipGetErrorString (error), error, __FILE__, __LINE__);    \
	  exit (EXIT_FAILURE);                                               \
	}                                                                    \
    } while (0)

__global__ void
kern ()
{
  while (1)
    __builtin_amdgcn_s_sleep (1);
}

int
main (int argc, char* argv[])
{
  kern<<<1, 1>>> ();
  CHECK (hipDeviceSynchronize ());
}

