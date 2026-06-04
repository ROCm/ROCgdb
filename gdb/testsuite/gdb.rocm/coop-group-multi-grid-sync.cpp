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

/* This program exercises a multi-device cooperative kernel launch via
   hipLaunchCooperativeKernelMultiDevice.  The kernel uses both
   cooperative_groups::this_grid ().sync () (intra-device GWS) and
   cooperative_groups::this_multi_grid ().sync () (cross-device sync).

   When run on a system with fewer than two GPUs that each support
   cooperative multi-device launch, the program prints a message and
   exits cleanly.  */

#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#include <stdio.h>
#include <stdlib.h>

#include "rocm-test-utils.h"

namespace cg = cooperative_groups;

constexpr int N_USED_GPUS = 2;
constexpr unsigned int N_PER_DEVICE = 256;
constexpr unsigned int group_size = 64;
constexpr unsigned int num_groups = 2;

/* Two-phase cooperative kernel running on every device.

   Phase 1: each thread writes (grid_rank + 1) * (i + 1) into its slot.
   GWS    : intra-device grid synchronization via this_grid ().sync ().
   MGWS   : cross-device synchronization via this_multi_grid ().sync ().
   Phase 2: thread 0 of each grid writes its partial sum into
   result[grid_rank + 1] (a slot in the shared host-coherent buffer).
   Then thread 0 of grid 0 aggregates the partials into result[0].  */

__global__ void
coop_multi_grid_sync_kernel (int *data, unsigned int n_elements,
			     long *result)
{
  cg::grid_group grid = cg::this_grid ();
  cg::multi_grid_group mgrid = cg::this_multi_grid ();

  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int stride = gridDim.x * blockDim.x;

  unsigned int grid_rank = mgrid.grid_rank ();

  /* Phase 1: each thread writes data into its slot.  */
  for (unsigned int i = tid; i < n_elements; i += stride)
    data[i] = (int) ((grid_rank + 1) * (i + 1));   /* before-grid-sync line */

  /* Intra-device grid sync (GWS).  */
  grid.sync ();			       /* grid-sync line */

  /* Each grid computes its partial sum and stores it in the shared
     buffer at slot (grid_rank + 1).  Slot 0 is reserved for the
     final total.  */
  if (tid == 0)
    {
      long sum = 0;
      for (unsigned int i = 0; i < n_elements; i++)
	sum += data[i];
      result[grid_rank + 1] = sum;     /* after-grid-sync line */
    }

  /* Cross-device sync (multi-grid GWS).  */
  mgrid.sync ();		       /* multi-grid-sync line */

  /* Grid 0 aggregates partial sums from all grids into result[0].  */
  if (grid_rank == 0 && tid == 0)
    {
      long total = 0;
      unsigned int n_grids = mgrid.num_grids ();
      for (unsigned int g = 0; g < n_grids; g++)
	total += result[g + 1];
      result[0] = total;	       /* after-multi-grid-sync line */
    }
}

int
main ()
{
  int n_devices = 0;
  CHECK (hipGetDeviceCount (&n_devices));
  if (n_devices < N_USED_GPUS)
    {
      printf ("Multi-device cooperative test needs >= %d GPUs"
	      " (found %d), skipping.\n", N_USED_GPUS, n_devices);
      return 0;
    }

  /* Pick the first N_USED_GPUS devices that support
     cooperativeMultiDeviceLaunch.  In a mixed-architecture system not
     every GPU has GWS support, so we ignore unsupported ones rather
     than failing the whole test.  */
  int selected[N_USED_GPUS];
  int n_gpus = 0;
  for (int id = 0; id < n_devices && n_gpus < N_USED_GPUS; id++)
    {
      hipDeviceProp_t p;
      CHECK (hipGetDeviceProperties (&p, id));
      if (p.cooperativeMultiDeviceLaunch)
	{
	  selected[n_gpus] = id;
	  n_gpus++;
	}
    }
  if (n_gpus < N_USED_GPUS)
    {
      printf ("Fewer than %d devices in 0..%d support cooperative"
	      " multi-device launch, skipping.\n",
	      N_USED_GPUS, n_devices - 1);
      return 0;
    }

  /* n-gpus-final line.  */
  int *data_d[N_USED_GPUS] = {};
  hipStream_t stream[N_USED_GPUS] = {};

  /* The shared result buffer is host-coherent so every device can
     read/write it without explicit memcpys.  Layout:
       result[0]            -- final total (written by grid 0 only)
       result[1 .. n_gpus]  -- per-grid partial sums.  */
  long *result_h = nullptr;

  for (int i = 0; i < n_gpus; i++)
    {
      CHECK (hipSetDevice (selected[i]));
      CHECK (hipMalloc ((void **) &data_d[i], N_PER_DEVICE * sizeof (int)));
      CHECK (hipMemset (data_d[i], 0, N_PER_DEVICE * sizeof (int)));
      CHECK (hipStreamCreate (&stream[i]));
    }

  CHECK (hipSetDevice (selected[0]));
  CHECK (hipHostMalloc ((void **) &result_h, (n_gpus + 1) * sizeof (long),
			hipHostMallocCoherent));
  for (int i = 0; i < n_gpus + 1; i++)
    result_h[i] = 0;

  /* Build per-device launch parameters.  Each device gets the same
     grid/block dims and a pointer to *its own* data buffer, but the
     result pointer is the single shared host-coherent buffer.  */
  hipLaunchParams launch_params[N_USED_GPUS];
  void *args[N_USED_GPUS][3];
  unsigned int n_elements = N_PER_DEVICE;

  for (int i = 0; i < n_gpus; i++)
    {
      args[i][0] = (void *) &data_d[i];
      args[i][1] = (void *) &n_elements;
      args[i][2] = (void *) &result_h;

      launch_params[i].func
	= reinterpret_cast<void *> (coop_multi_grid_sync_kernel);
      launch_params[i].gridDim = dim3 (num_groups, 1, 1);
      launch_params[i].blockDim = dim3 (group_size, 1, 1);
      launch_params[i].sharedMem = 0;
      launch_params[i].stream = stream[i];
      launch_params[i].args = args[i];
    }

  CHECK (hipLaunchCooperativeKernelMultiDevice (launch_params, n_gpus, 0));

  for (int i = 0; i < n_gpus; i++)
    {
      CHECK (hipSetDevice (selected[i]));
      CHECK (hipDeviceSynchronize ());
    }

  long total = result_h[0];

  for (int i = 0; i < n_gpus; i++)
    {
      CHECK (hipSetDevice (selected[i]));
      CHECK (hipFree (data_d[i]));
      CHECK (hipStreamDestroy (stream[i]));
    }
  CHECK (hipSetDevice (selected[0]));
  CHECK (hipHostFree (result_h));

  /* Compute the expected total: for grid g (0-based), the thread that
     writes slot i stores (g + 1) * (i + 1).  Per grid, the sum is
     (g + 1) * N_PER_DEVICE * (N_PER_DEVICE + 1) / 2.  */
  long per_grid = (long) N_PER_DEVICE * (N_PER_DEVICE + 1) / 2;
  long expected = 0;
  for (int g = 0; g < n_gpus; g++)
    expected += (long) (g + 1) * per_grid;

  if (total != expected)
    {
      fprintf (stderr, "FAILED: total %ld, expected %ld\n", total, expected);
      return EXIT_FAILURE;
    }

  return EXIT_SUCCESS;
}
