/* Copyright 2023-2024 Free Software Foundation, Inc.
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
#include <iostream>
#include <limits>

#define CHECK(cmd)							\
  {									\
    hipError_t error = cmd;						\
    if (error != hipSuccess)						\
      {									\
	fprintf (stderr, "error: '%s'(%d) at %s:%d\n",			\
		 hipGetErrorString (error), error, __FILE__, __LINE__);	\
	exit (EXIT_FAILURE);						\
      }									\
  }

__device__ void
enable_alu_exceptions ()
{
  /* By default, ALU exceptions are not enabled.  Break here and use the
     debugger to enable exceptions.  */
}

__global__ void
raise_invalid ()
{
  enable_alu_exceptions ();
  volatile float a = -1;
  float b = sqrt (a); /* Break here for invalid.  */
}

__global__ void
raise_denorm ()
{
  enable_alu_exceptions ();
  volatile float inp = std::numeric_limits<float>::denorm_min ();
  float b = inp * inp; /* Break here for denorm.  */
}

__global__ void
raise_float_div0 ()
{
  enable_alu_exceptions ();
  volatile float a = 1, b = 0;
  float c = a / b; /* Break here for float_div0.  */
}

__global__ void
raise_overflow ()
{
  enable_alu_exceptions ();
  volatile float max = std::numeric_limits<float>::max ();
  float b = max * 2.0f; /* Break here for overflow.  */
}

__global__ void
raise_underflow ()
{
  enable_alu_exceptions ();
  volatile float min = std::numeric_limits<float>::min ();
  float c = min * min; /* Break here for underflow.  */
}

__global__ void
raise_inexact ()
{
  enable_alu_exceptions ();
  volatile float f = 1.1f;
  float r = f * f; /* Break here for inexact.  */
}

__global__ void
raise_int_div0 ()
{
  enable_alu_exceptions ();
  volatile int a = 1, b = 0;
  int c = a / b; /* Break here for int_div0.  */
}

int
main (int argc, char **argv)
{
  if (argc != 2)
    {
      std::cerr
	<< "Usage: " << argv[0]
	<< (" invalid|denorm|float_div0|overflow|underflow|inexact|"
	    "int_div0") << std::endl;
      return EXIT_FAILURE;
    }

  const std::string test (argv[1]);
  if (test == "invalid")
    raise_invalid<<<1, 1>>> ();
  else if (test == "denorm")
    raise_denorm<<<1, 1>>> ();
  else if (test == "float_div0")
    raise_float_div0<<<1, 1>>> ();
  else if (test == "overflow")
    raise_overflow<<<1, 1>>> ();
  else if (test == "underflow")
    raise_underflow<<<1, 1>>> ();
  else if (test == "inexact")
    raise_inexact<<<1, 1>>> ();
  else if (test == "int_div0")
    raise_int_div0<<<1, 1>>> ();
  else
    {
      std::cerr << "Unsupported test " << test << std::endl;
      return EXIT_FAILURE;
    }

  CHECK (hipDeviceSynchronize ());
  return EXIT_SUCCESS;
}
