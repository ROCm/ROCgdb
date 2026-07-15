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

/* Run a HIP program where workgroups are organized in clusters.  */

#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include <stdio.h>
#include <stdlib.h>

#include "rocm-test-utils.h"

__global__ void
#ifdef WITH_CLUSTER
__cluster_dims__(3, 2, 1)
#endif
kernel ()
{
  namespace cg = cooperative_groups;
  cg::cluster_group c = cg::this_cluster();

  /* Index of the calling block within cluster.  */
  auto block_in_cluster = c.block_index ();

  __builtin_amdgcn_s_sleep (1); /* break-here */
}

int
main ()
{
  kernel<<<dim3 (9, 4, 2), dim3 (30, 5, 1)>>>();
  CHECK (hipDeviceSynchronize ());
  return 0;
}
