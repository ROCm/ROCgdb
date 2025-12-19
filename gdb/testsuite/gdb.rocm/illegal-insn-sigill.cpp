/* Copyright 2023-2024 Free Software Foundation, Inc.
   Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.

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
#include <cstdlib>

__global__ void
illegal_insn ()
{
  /* 0xdeadbeef is a known invalid instruction encoding.  */
  asm (".byte 0xef, 0xbe, 0xad, 0xde");
}


int
main (int argc, char **argv)
{
  illegal_insn<<<1, 1>>> ();
  hipError_t err = hipDeviceSynchronize ();
  if (err == hipErrorLaunchFailure)
    {
      /* Depending on the system configuration, the HIP runtime might or might
	 not call abort(3) when it receives the GPU error.  Make sure to call
	 it ourself so the testcase can match the SIGABRT.  */
      abort ();
    }
  return (err == hipSuccess) ? EXIT_SUCCESS : EXIT_FAILURE;
}
