/* Copyright 2021-2022 Free Software Foundation, Inc.
   Copyright (C) 2021-2022 Advanced Micro Devices, Inc. All rights reserved.

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

  /* This first divergence is here so GDB can check which lane is the
     'then' and which is the 'else' lane, based on which lanes become
     active/inactive.  This if/then/else block is stepped with "maint
     set lane-divergence-support off".  */
  if (gid % 2)					/* if_0_cond */
    elem = const_array[gid] + 1;		/* if_0_then */
  else
    elem = const_array[gid] + 3;		/* if_0_else */

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

__global__ void
kernel (const int *in, struct test_struct *out)
{
  unsigned gid0 = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
  lane_pc_test (gid0, in, out);
}

int
main ()
{
  dim3 grid_dim (1);
  dim3 block_dim (32);

  struct test_struct *sOutBuff;
  int *sInBuff;
  CHECK (hipMalloc (&sOutBuff, sizeof (struct test_struct)));
  CHECK (hipMalloc (&sInBuff, sizeof (int)
		    * (grid_dim.x * grid_dim.y * grid_dim.z
		       * block_dim.x
		       * block_dim.y
		       * block_dim.z)));

  hipLaunchKernelGGL (kernel, grid_dim, block_dim, 0, 0, sInBuff, sOutBuff);

  hipDeviceSynchronize ();

  return 0;
}
