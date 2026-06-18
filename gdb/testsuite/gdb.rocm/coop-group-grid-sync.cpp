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

/* This program exercises a single-device cooperative kernel launch.
   The kernel uses cooperative_groups::this_grid ().sync (), which is
   implemented on AMD GPUs using the Global Wave Sync (GWS) hardware
   mechanism.  It is launched through hipLaunchCooperativeKernel so
   the whole grid is co-resident and can synchronize at the grid
   level.

   The companion .exp file uses this program to exercise the debugger
   while a kernel is running with GWS-based grid synchronization.  */

#include <hip/hip_cooperative_groups.h>
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include "rocm-test-utils.h"

namespace cg = cooperative_groups;

/* Use small but non-trivial launch dimensions so the test runs quickly
   while still creating waves from multiple workgroups participating
   in the GWS barrier.  Two workgroups of 64 threads each give 128
   threads in total.

   N is exactly one slot per thread so Phase 2 has no inter-thread
   write conflict and the host-side expected values can be computed
   straightforwardly.  */

constexpr unsigned int group_size = 64;
constexpr unsigned int num_groups = 2;
constexpr unsigned int total_threads = group_size * num_groups;
constexpr unsigned int N = total_threads;

/* Two-phase grid-cooperative kernel.

   Phase 1: every thread writes its slot in in_buf.
   GWS    : the whole grid synchronizes via cooperative_groups::this_grid ().
   Phase 2: every thread reads a slot owned by a thread in a *different*
   workgroup and stores it into out_buf.

   Because the Phase 2 read targets a slot written by a different
   workgroup during Phase 1, this is only correct if grid.sync ()
   actually synchronized every wave in the grid.  */

__global__ void
coop_grid_sync_kernel (int *in_buf, int *out_buf)
{
  cg::grid_group grid = cg::this_grid ();

  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

  /* Phase 1: each thread writes its own value.  */
  in_buf[tid] = (int) (tid + 1);	/* before-sync line */

  /* Grid-wide synchronization (GWS).  */
  grid.sync ();				/* sync line */

  /* Phase 2: read the slot owned by a thread in a different workgroup
     and store it in our own slot of the output buffer.  */
  unsigned int peer = (tid + blockDim.x) % total_threads;
  int v = in_buf[peer];			/* after-sync line */
  out_buf[tid] = v;
}

int
main ()
{
  int n_devices = 0;
  CHECK (hipGetDeviceCount (&n_devices));
  if (n_devices <= 0)
    {
      printf ("No HIP devices found, skipping.\n");
      return 0;
    }

  int device_id = -1;
  hipDeviceProp_t props;
  for (int id = 0; id < n_devices; ++id)
    {
      CHECK (hipGetDeviceProperties (&props, id));
      if (props.cooperativeLaunch)
	{
	  device_id = id;
	  break;
	}
    }

  if (device_id < 0)
    {
      printf ("None of the %d HIP device(s) support cooperative launch, "
	      "skipping.\n",
	      n_devices);
      return 0;
    }

  CHECK (hipSetDevice (device_id));

  int *in_d = nullptr;
  int *out_d = nullptr;
  CHECK (hipMalloc ((void **) &in_d, N * sizeof (int)));
  CHECK (hipMalloc ((void **) &out_d, N * sizeof (int)));
  CHECK (hipMemset (in_d, 0, N * sizeof (int)));
  CHECK (hipMemset (out_d, 0, N * sizeof (int)));

  dim3 grid_dim (num_groups, 1, 1);
  dim3 block_dim (group_size, 1, 1);

  void *kernel_args[2];
  kernel_args[0] = (void *) &in_d;
  kernel_args[1] = (void *) &out_d;

  /* Launch the kernel cooperatively.  This is the API that enables
     grid.sync () support via GWS.  The trailing 0, 0 arguments are
     sharedMem and stream respectively.  */
  CHECK (hipLaunchCooperativeKernel
	   (reinterpret_cast<void *> (coop_grid_sync_kernel),
	    grid_dim, block_dim, kernel_args, 0, 0));

  CHECK (hipDeviceSynchronize ());

  int out_h[N];
  CHECK (hipMemcpy (out_h, out_d, N * sizeof (int), hipMemcpyDeviceToHost));

  CHECK (hipFree (in_d));
  CHECK (hipFree (out_d));

  /* Each thread tid stores in_buf[(tid + group_size) % total_threads]
     into out_buf[tid].  After Phase 1, in_buf[k] == k + 1, so the
     expected value at out_h[tid] is ((tid + group_size) %
     total_threads) + 1.  */
  int errors = 0;
  for (unsigned int tid = 0; tid < N; tid++)
    {
      unsigned int peer = (tid + group_size) % total_threads;
      int expected = (int) (peer + 1);
      if (out_h[tid] != expected)
	{
	  fprintf (stderr, "mismatch at %u: got %d, expected %d\n",
		   tid, out_h[tid], expected);
	  errors++;
	}
    }

  if (errors != 0)
    {
      fprintf (stderr, "FAILED: %d mismatches\n", errors);
      return EXIT_FAILURE;
    }

  return EXIT_SUCCESS;
}
