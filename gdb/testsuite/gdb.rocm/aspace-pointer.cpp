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

/* Run a HIP program with pointers to particular address spaces.  */

#include <hip/hip_runtime.h>
#include "gdb_watchdog.h"
#include "rocm-test-utils.h"

/* Address space values are defined by the ABI.  */
#define LOCAL __attribute__((address_space(3)))
#define PRIVATE_LANE __attribute__((address_space(5)))

__device__ short global_var = 1234;

__global__ void
kernel ()
{
  __shared__ short local_var1;
  __shared__ short local_var2;
  local_var1 = 101;
  local_var2 = 102;

  short priv_var1 = 41;
  short priv_var2 = 42;

  short *generic_ptr0 = nullptr;
  short *generic_ptr1 = &global_var;
  short *generic_ptr2 = &local_var1;
  short *generic_ptr3 = &priv_var1;

  LOCAL short *local_ptr0 = nullptr;
  LOCAL short *local_ptr1 = (LOCAL short *) &local_var2;
  LOCAL short *local_ptr2 = (LOCAL short *) generic_ptr2;

  PRIVATE_LANE short *priv_ptr0 = nullptr;
  PRIVATE_LANE short *priv_ptr1 = (PRIVATE_LANE short *) &priv_var2;
  PRIVATE_LANE short *priv_ptr2 = (PRIVATE_LANE short *) generic_ptr3;

  int sizeof_generic_ptr = sizeof(generic_ptr0);
  int sizeof_local_ptr = sizeof(local_ptr0);
  int sizeof_priv_ptr = sizeof(priv_ptr0);

  NOP (1); /* break-here */
}

int
main ()
{
  kernel<<<5, 200>>> ();
  CHECK (hipDeviceSynchronize ());
  return 0;
}
