/* Copyright (C) 2021-2026 Advanced Micro Devices, Inc. All rights reserved.

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
#include <stdlib.h>
#include <unistd.h>
#include "rocm-test-utils.h"

/* Occupy wave slots until the process is killed.  Thread 0 of each
   workgroup increments STARTED so the host can wait until the slots
   are actually held.

   Workgroups must be launched at maxThreadsPerBlock.  A hog made of
   one-wave workgroups is easily displaced by another large workload.  */

__global__ void
hog_kernel (int *started)
{
  if (threadIdx.x == 0)
    atomicAdd_system (started, 1);
  while (true)
    __builtin_amdgcn_s_sleep (1);
}

/* ARGV[1], if given, is how many seconds to wait for every hog
   workgroup to start before giving up.  */

int
main (int argc, char **argv)
{
  int ready_secs = 10;
  if (argc > 1)
    ready_secs = atoi (argv[1]);
  if (ready_secs < 1)
    ready_secs = 1;

  int deviceId;
  CHECK (hipGetDevice (&deviceId));

  hipDeviceProp_t props;
  CHECK (hipGetDeviceProperties (&props, deviceId));

  int max_waves = device_max_waves (&props);
  int waves_per_wg
    = (props.warpSize > 0
       ? (props.maxThreadsPerBlock + props.warpSize - 1) / props.warpSize
       : 0);
  if (max_waves <= 0 || waves_per_wg <= 0)
    {
      fprintf (stderr, "error: could not determine device wave capacity\n");
      return EXIT_FAILURE;
    }

  /* Occupy a quarter of the device, leaving room for other work.  */
  int nwg = (max_waves / 4) / waves_per_wg;
  if (nwg < 1)
    nwg = 1;
  int hog_waves = nwg * waves_per_wg;

  int *host_started;
  CHECK (hipHostMalloc ((void **) &host_started, sizeof (int),
			hipHostMallocMapped | hipHostMallocCoherent));
  __atomic_store_n (host_started, 0, __ATOMIC_RELAXED);

  int *dev_started;
  CHECK (hipHostGetDevicePointer ((void **) &dev_started, host_started, 0));

  hog_kernel<<<nwg, props.maxThreadsPerBlock>>> (dev_started);
  CHECK (hipGetLastError ());

  /* Wait until every hog workgroup has actually started.  */
  for (int i = 0;
       i < ready_secs * 100
       && __atomic_load_n (host_started, __ATOMIC_RELAXED) < nwg;
       i++)
    usleep (10000);
  int started_wg = __atomic_load_n (host_started, __ATOMIC_RELAXED);
  if (started_wg < nwg)
    {
      fprintf (stderr,
	       "error: hog parked %d of %d waves (device_max=%d)\n",
	       started_wg * waves_per_wg, hog_waves, max_waves);
      return EXIT_FAILURE;
    }

  printf ("hog ready: parked %d waves (device_max=%d)\n", hog_waves,
	  max_waves);
  fflush (stdout);

  CHECK (hipDeviceSynchronize ());
  return 0;
}
