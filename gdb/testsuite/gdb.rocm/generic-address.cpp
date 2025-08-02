/* This testcase is part of GDB, the GNU debugger.

   Copyright (C) 2026 Free Software Foundation, Inc.
   Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

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
#include <cassert>

#include "rocm-test-utils.h"

__global__ void
kern ()
{
  int lane = threadIdx.x;
  volatile int local = lane;
  alignas (4) volatile unsigned char arr[5] = {0x11, 0x22, 0x33, 0x44, 0x55};
  volatile int *gen_local_addr = &local;
  /* A generic address to a 32-bit integer that is straddled over
     two DWORDS in the backing memory.  */
  volatile uint32_t *gen_arr1_addr = (uint32_t *) &arr[1];

  /* Break here 1; debugger's actions:
     - Switch to lane 11.
     - Run some checks.
     - Change "local" to 1337.  */
  NOP (1);

  if (lane == 11)
    {
      assert (local == 1337);
      /* Break here 2: If hit, the assert must have gone smoothly.  */
      NOP (2);
    }

  /* Break here 3; debugger's actions:
     - Check if still in lane 11.
     - Run some other checks.
     - Change "arr[1..4]" to 0x87654321.  */
  NOP (3);

  if (lane == 11)
    {
      assert (*((uint32_t *) &arr[1]) == 0x87654321);
      /* Break here 4: If hit, the assert must have gone smoothly.  */
      NOP (4);
    }
}

int
main()
{
  int device_id;
  hipDeviceProp_t props;
  CHECK (hipGetDevice (&device_id));
  CHECK (hipGetDeviceProperties (&props, device_id));

  /* There must be at least 12 lanes.  */
  assert (props.warpSize > 11);

  kern<<<1, props.warpSize>>> ();

  return hipDeviceSynchronize () != hipSuccess;
}
