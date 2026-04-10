/* This testcase is part of GDB, the GNU debugger.

   Copyright 2025-2026 Free Software Foundation, Inc.

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
#include "rocm-test-utils.h"

__global__ void
kern (int *ptr)
{
  (*ptr)++;
}

int *managed_var = nullptr;

int
main ()
{
  int device_id = 0;

  /* Allocate a variable using managed memory so that we can allocate
     the variable in the 256MB aperture on non-ReBar systems.  A
     __device__ global would only be guaranteed to be in the aperture
     on ReBar systems.  */
  CHECK (hipMallocManaged (&managed_var, sizeof (int)));

  /* Apply hints before any access.  On non-ReBar systems, force the
     driver to find a spot on the GPU, CPU-visible in the 256MB BAR
     aperture.  */
  CHECK (hipMemAdvise (managed_var, sizeof (int),
		       hipMemAdviseSetAccessedBy, hipCpuDeviceId));
  CHECK (hipMemAdvise (managed_var, sizeof (int),
		       hipMemAdviseSetPreferredLocation, device_id));

  /* Ensure the physical pages are on the GPU.  */
  CHECK (hipMemPrefetchAsync (managed_var, sizeof (int), device_id, 0));

  /* Warm up the GPU.  */
  kern<<<1, 1>>> (managed_var);
  CHECK (hipDeviceSynchronize ());

  printf ("Pointer address: %p\n", (void *) managed_var);
  printf ("Attempting host write...\n");

  /* Now update the device global from the host.  On ReBar systems,
     and on non-ReBar systems (if the hints worked), this writes
     across the PCIe BAR into the GPU's memory controller.  */
  *managed_var = 8;

  printf ("Write successful.  Value: %d\n", *managed_var);

  CHECK (hipFree (managed_var));
  return 0;
}
