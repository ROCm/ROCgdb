/* Copyright 2020-2022 Free Software Foundation, Inc.
   Copyright (C) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.

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

#define CHECK(cmd)					\
  do							\
    {							\
      hipError_t error = cmd;				\
      if (error != hipSuccess)				\
	{						\
	  fprintf(stderr, "error: '%s'(%d) at %s:%d\n",	\
		  hipGetErrorString (error), error,	\
		  __FILE__, __LINE__);			\
	  exit (EXIT_FAILURE);				\
	}						\
    } while (0)

struct test_struct
{
  int int_elem;
  char char_elem;
  int array_elem[32];
};

__device__ const struct test_struct const_struct = {
  32, '2',
  {
    32, 31, 30, 29, 28, 27, 26, 25,
    24, 23, 22, 21, 20, 19, 18, 17,
    16, 15, 14, 13, 12, 11, 10, 9,
    8, 7, 6, 5, 4, 3, 2, 1
  }
};

__device__ const int const_array[32] = {
  1, 1, 1, 1, 1, 5, 5, 7,
  2, 2, 2, 2, 2, 5, 5, 10,
  3, 3, 3, 3, 3, 5, 5, 2,
  4, 4, 4, 4, 4, 5, 5, 3
};

__device__ static unsigned
get_global_id (int dim_idx)
{
  switch (dim_idx)
    {
    case 0:
      return blockIdx.x * blockDim.x + threadIdx.x;
    case 1:
      return blockIdx.y * blockDim.y + threadIdx.y;
    case 2:
      return blockIdx.z * blockDim.z + threadIdx.z;
    }

  abort ();
  __builtin_unreachable ();
}

__device__ int
bar (int val)
{
  return val;
}

__device__ int
foo (int val)
{
  return bar (val);
}

__device__ void
lane_pc_test (unsigned gid, const int *in, struct test_struct *out)
{
  int elem;

  out->int_elem = const_struct.int_elem;
  out->char_elem = const_struct.char_elem;

  if (gid % 2)					/* if_1_cond */
    {
      elem = const_array[gid] + 1;		/* if_1_then */
      elem = const_array[gid] + 2;		/* if_1_then_2 */
    }
  else
    {
      elem = const_array[gid] + 3;		/* if_1_else */
      elem = const_array[gid] + 4;		/* if_1_else_2 */
    }
						/* if_1_end  */

  if (gid % 2)					/* if_2_cond */
    elem = foo (const_array[gid]);		/* if_2_then */
  else
    elem = const_array[gid];			/* if_2_else */
						/* if_2_end  */

  /* This condition is always false.  */
  if (gid == -1)				/* if_3_cond */
    elem = const_array[gid] + 1;		/* if_3_then */
  else
    elem = const_array[gid] + 2;		/* if_3_else */
						/* if_3_end  */

  atomicAdd (&out->int_elem, elem);
  out->array_elem[gid] = elem;
}

__device__ void
loclistx_test (const int *in, struct test_struct *out)
{
  unsigned gid = get_global_id (0);
  int elem;

  out->int_elem = const_struct.int_elem;
  out->char_elem = const_struct.char_elem;

  if (gid % 4)
    elem = const_array[gid] + 1;
  else
    elem = const_array[gid] + 2;

  if (gid % 4)
    elem = foo (const_array[gid]);
  else
    elem = const_array[gid];

  atomicAdd (&out->int_elem, elem);
  out->array_elem[gid] = elem;
}

__global__ void
kernel (const int *in, struct test_struct *out)
{
  unsigned gid0 = get_global_id (0);
  lane_pc_test (gid0, in, out);
  loclistx_test (in, out);
}

int
main ()
{
  dim3 blocks (1);
  dim3 threadsPerBlock (32);

  struct test_struct *sOutBuff;
  int *sInBuff;
  CHECK (hipMalloc (&sOutBuff, sizeof (struct test_struct)));
  CHECK (hipMalloc (&sInBuff, sizeof (int)
		    * (blocks.x * blocks.y * blocks.z
		       * threadsPerBlock.x
		       * threadsPerBlock.y
		       * threadsPerBlock.z)));

  hipLaunchKernelGGL (kernel, blocks, threadsPerBlock, 0, 0,
		      sInBuff, sOutBuff);

  /* Wait until kernel finishes.  */
  CHECK (hipDeviceSynchronize ());

  return 0;
}
