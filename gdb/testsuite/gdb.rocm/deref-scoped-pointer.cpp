/* Copyright 2023 Free Software Foundation, Inc.
   Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.

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

__device__ static void
done ()
{
}

/* This is initialized by the kernel function below.  */
__device__ size_t *global_ptr = nullptr;

/* This is initialized by GDB.  */
__device__ size_t *global_ptr2 = nullptr;

__global__ void
kernel ()
{
  size_t local_var = blockIdx.x * blockDim.x + threadIdx.x;

  /* This will be zero for the first HIP thread, or lane.  */
  if (local_var == 0)
    global_ptr = &local_var;

  /* Ensure thread 0 has assigned to global_ptr before the other
     thread attempts to read it.  Even though the lanes are in the
     same wave, this prevents the compiler from restructuring the
     control flow graph in a way that would break the test.  */
  __syncthreads ();

  /* This is just to confirm how dereferencing the pointer from
     different lanes yield a different value.  GDB should behave the
     same.  */
  assert (global_ptr != nullptr);
  printf ("global_ptr=%p, *global_ptr=%lx\n", global_ptr, *global_ptr);

  done (); /* set breakpoint here */

  /* Convenience nullptr check useful if you run the program manually
     or skip part of the testcase.  */
  if (global_ptr2 != nullptr)
    {
      printf ("global_ptr2=%p, *global_ptr2=%lx\n", global_ptr2, *global_ptr2);
      /* GDB made global_ptr2 point to &local_var too.  If GDB wrote
	 to global_ptr2 correctly, this should pass.  */
      assert (global_ptr == global_ptr2);
    }
  else
    printf ("global_ptr2=nullptr\n");

}

int
main ()
{
  /* We only need more than one lane.  Two is sufficient.  */
  kernel<<<dim3 (1), dim3 (2)>>> ();
  CHECK (hipDeviceSynchronize ());
  return 0;
}
