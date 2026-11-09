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

/* Launch a kernel on every visible GPU before any of them finishes, so
   the debugger sees all of them live at once.  */

#include <hip/hip_runtime.h>
#include "rocm-test-utils.h"

/* Waves per GPU.  A 32-thread block fits in one wave on every supported
   GPU.  */
#define WAVES 4
#define BLOCK_DIM 32

__global__ void
work_kernel (volatile int *hold, volatile int *value)
{
  /* Only the first wave hits the breakpoints.  The others keep running,
     so info threads has more than one wave per GPU.  */
  if (blockIdx.x == 0 && threadIdx.x == 0)
    *value += 1;		/* Break before spin.  */

  while (*hold)
    {
      if (blockIdx.x == 0 && threadIdx.x == 0)
	*value += 1;		/* Break in spin.  */
      WAIT_MEM;
      __builtin_amdgcn_s_sleep (8);
    }
}

/* Inspected by the test.  The arrays have one element per visible GPU.
   Pointers are declared "volatile" to prevent the optimizer from
   eliminating their storage and keeping the values in registers
   instead.  num_devices stays a plain int because its address is
   passed to hipGetDeviceCount, which takes a plain "int *".  */
int num_devices;
/* (domain << 16) | (bus << 8) | device.  */
unsigned int *volatile pci_location;
/* Device memory of each GPU.  *bufs[dev] starts at 100 + dev.  */
volatile int **volatile bufs;
/* Read by every kernel.  The test clears it to let the kernels exit.  */
volatile int *volatile hold;

int
main ()
{
  CHECK (hipGetDeviceCount (&num_devices));

  pci_location = new unsigned int[num_devices];
  bufs = new volatile int *[num_devices];

  for (int dev = 0; dev < num_devices; ++dev)
    {
      hipDeviceProp_t props;
      CHECK (hipGetDeviceProperties (&props, dev));
      pci_location[dev] = ((props.pciDomainID << 16) | (props.pciBusID << 8)
			   | props.pciDeviceID);
    }

  /* hipHostMalloc writes through a plain pointer.  Assigning the
     result keeps the volatile qualifier on hold.  */
  void *raw;
  CHECK (hipHostMalloc (&raw, sizeof (int), hipHostMallocCoherent));
  hold = (volatile int *) raw;
  *hold = 1;

  /* Break before launch.  */
  hipStream_t *streams = new hipStream_t[num_devices];
  for (int dev = 0; dev < num_devices; ++dev)
    {
      int initial = 100 + dev;

      CHECK (hipSetDevice (dev));
      CHECK (hipMalloc ((void **) &bufs[dev], sizeof (int)));
      CHECK (hipMemcpy ((void *) bufs[dev], &initial, sizeof (int),
			hipMemcpyHostToDevice));

      /* Keep the kernel off the null stream.  It keeps running, and a
	 later null-stream call would otherwise wait for it.  */
      CHECK (hipStreamCreateWithFlags (&streams[dev], hipStreamNonBlocking));
      work_kernel<<<WAVES, BLOCK_DIM, 0, streams[dev]>>> (hold, bufs[dev]);
      CHECK (hipGetLastError ());
    }

  /* Break after launch.  */
  while (*hold)
    ;

  int ret = 0;
  for (int dev = 0; dev < num_devices; ++dev)
    {
      int value;

      CHECK (hipSetDevice (dev));
      CHECK (hipStreamSynchronize (streams[dev]));
      CHECK (hipStreamDestroy (streams[dev]));
      CHECK (hipMemcpy (&value, (void *) bufs[dev], sizeof (int),
			hipMemcpyDeviceToHost));
      CHECK (hipFree ((void *) bufs[dev]));
      /* The first wave adds one before it enters the spin.  */
      if (value < 101 + dev)
	ret = 1;
    }

  CHECK (hipHostFree ((void *) hold));
  delete[] pci_location;
  delete[] bufs;
  delete[] streams;
  return ret;
}
