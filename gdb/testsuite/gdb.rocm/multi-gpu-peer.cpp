/* This testcase is part of GDB, the GNU debugger.

   Copyright 2026 Free Software Foundation, Inc.

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

/* For each visible GPU, allocate memory on that GPU and dispatch a
   kernel that writes it on the next GPU, wrapping around to GPU 0.  */

#include <hip/hip_runtime.h>
#include "rocm-test-utils.h"

#define INIT_VALUE 7

__global__ void
warmup_kernel ()
{
}

/* Stays running on the allocation GPU so its PC can be read while the
   compute GPU is stopped.  */

__global__ void
wait_kernel (volatile int *hold)
{
  while (*hold)
    {
      WAIT_MEM;
      __builtin_amdgcn_s_sleep (8);
    }
}

__global__ void
store_kernel (int *val)
{
  *val += 10;  /* Break in kernel.  */
  WAIT_MEM;
  *val += 20;
  WAIT_MEM;

  /* Some devices that don't support "precise memory" miss watchpoints when
     they would trigger near the end of the kernel.  Execute a bunch of sleeps
     to make sure this doesn't happen.  */
  for (int i = 0; i < 100000; ++i)
    __builtin_amdgcn_s_sleep (8);
}

/* Inspected by the test.  The arrays have one element per visible GPU.  */
int num_devices;
unsigned int *pci_location;	/* (domain << 16) | (bus << 8) | device.  */
int *can_access;		/* Can the compute GPU access this memory?  */
int *buf;			/* Memory allocated on alloc_dev.  */
int *hold;			/* Wait flag on the allocation GPU.  */
int alloc_dev = -1;

int
main ()
{
  CHECK (hipGetDeviceCount (&num_devices));

  pci_location = new unsigned int[num_devices];
  can_access = new int[num_devices] ();

  for (int dev = 0; dev < num_devices; ++dev)
    {
      hipDeviceProp_t props;
      CHECK (hipGetDeviceProperties (&props, dev));
      pci_location[dev] = ((props.pciDomainID << 16) | (props.pciBusID << 8)
			   | props.pciDeviceID);
    }

  /* hipDeviceCanAccessPeer asks whether the compute GPU can read and
     write the allocation GPU.  */
  if (num_devices >= 2)
    {
      for (int dev = 0; dev < num_devices; ++dev)
	CHECK (hipDeviceCanAccessPeer (&can_access[dev],
				       (dev + 1) % num_devices, dev));
    }

  /* Break after device query.  */
  int ret = 0;
  for (int dev = 0; dev < num_devices; ++dev)
    {
      if (!can_access[dev])
	continue;

      int compute_dev = (dev + 1) % num_devices;
      int value = INIT_VALUE;

      alloc_dev = dev;
      CHECK (hipSetDevice (alloc_dev));
      CHECK (hipMalloc (&buf, sizeof (int)));
      CHECK (hipMemcpy (buf, &value, sizeof (int), hipMemcpyHostToDevice));

      CHECK (hipSetDevice (compute_dev));
      CHECK (hipDeviceEnablePeerAccess (alloc_dev, 0));

      /* Make sure the compute device's queue is mapped before the test
	 inserts watchpoints.  */
      warmup_kernel<<<1, 1>>> ();
      CHECK (hipDeviceSynchronize ());

      int hold_val = 1;
      CHECK (hipSetDevice (alloc_dev));
      CHECK (hipMalloc (&hold, sizeof (int)));
      CHECK (hipMemcpy (hold, &hold_val, sizeof (int),
			hipMemcpyHostToDevice));

      /* Break before dispatch.  */
      wait_kernel<<<1, 1>>> (hold);
      CHECK (hipSetDevice (compute_dev));
      store_kernel<<<1, 1>>> (buf);
      CHECK (hipDeviceSynchronize ());

      hold_val = 0;
      CHECK (hipSetDevice (alloc_dev));
      CHECK (hipMemcpy (hold, &hold_val, sizeof (int),
			hipMemcpyHostToDevice));
      CHECK (hipDeviceSynchronize ());
      CHECK (hipFree (hold));
      hold = nullptr;

      CHECK (hipMemcpy (&value, buf, sizeof (int), hipMemcpyDeviceToHost));
      CHECK (hipFree (buf));
      buf = nullptr;
      if (value != INIT_VALUE + 30)
	{
	  fprintf (stderr,
		   "unexpected value %d in GPU %d memory, kernel on GPU %d\n",
		   value, alloc_dev, compute_dev);
	  ret = 1;
	}
    }

  delete[] pci_location;
  delete[] can_access;
  return ret;
}
