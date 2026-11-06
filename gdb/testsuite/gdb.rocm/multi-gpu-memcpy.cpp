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

/* For each visible GPU, copy memory from that GPU to the next GPU,
   wrapping around to GPU 0, with hipMemcpyPeer, hipMemcpyPeerAsync,
   hipMemcpy2D, and hipMemcpy3DPeer.  */

#include <hip/hip_runtime.h>
#include "rocm-test-utils.h"

#define ROWS 8
#define COLS 8
#define COUNT (ROWS * COLS)
#define BUMP 1000

/* Flags in host-coherent memory, shared by the host and the kernels.  */

struct control
{
  int hold;
  int gate;
  int src_started;
  int dst_started;
  int gate_started;
};

__global__ void
warmup_kernel ()
{
}

__device__ void
hold_wave (volatile int *hold, volatile int *started)
{
  *started = 1;
  __threadfence_system ();
  while (*hold)
    {
      WAIT_MEM;
      __builtin_amdgcn_s_sleep (8);
    }
}

/* Stay running on the source and destination GPUs so that memory can
   be read from both while the host is stopped.  */

__global__ void
src_kernel (volatile int *hold, volatile int *started)
{
  hold_wave (hold, started);
}

__global__ void
dst_kernel (volatile int *hold, volatile int *started)
{
  hold_wave (hold, started);
}

/* Runs ahead of the async copy on its stream, so the copy stays queued
   until the host clears the gate.  */

__global__ void
gate_kernel (volatile int *gate, volatile int *started)
{
  hold_wave (gate, started);
}

__global__ void
bump_kernel (int *val)
{
  *val += BUMP;
  WAIT_MEM;

  /* Some devices that don't support "precise memory" miss watchpoints when
     they would trigger near the end of the kernel.  Execute a bunch of sleeps
     to make sure this doesn't happen.  */
  for (int i = 0; i < 100000; ++i)
    __builtin_amdgcn_s_sleep (8);
}

/* Enable access from DEV to PEER's memory, leaving DEV as the current
   device.  The ring enables the same access twice when there are only
   two GPUs.  */

static void
enable_peer (int dev, int peer)
{
  CHECK (hipSetDevice (dev));
  hipError_t status = hipDeviceEnablePeerAccess (peer, 0);
  if (status == hipErrorPeerAccessAlreadyEnabled)
    (void) hipGetLastError ();
  else
    CHECK (status);
}

/* Inspected by the test.  The arrays have one element per visible GPU.
   Pointers are declared "volatile" to prevent the optimizer from
   eliminating their storage and keeping the values in registers
   instead.  num_devices stays a plain int because its address is
   passed to hipGetDeviceCount, which takes a plain "int *".  */
int num_devices;
/* (domain << 16) | (bus << 8) | device.  */
unsigned int *volatile pci_location;
/* Can this GPU and the next access each other?  */
int *volatile can_access;
volatile int src_dev = -1;
/* src[i] is (src_dev + 1) * 100 + i.  */
int *volatile src;
int *volatile dst;
int *volatile dst_async;
/* Row 1 of the 2D copy's destination.  */
int *volatile dst_2d_row;
/* First element of row 1 of slice 1 of the 3D copy's destination.  */
int *volatile dst_3d_elem;
volatile hipError_t query_in_flight = hipSuccess;
volatile hipError_t query_done = hipErrorNotReady;

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

  if (num_devices >= 2)
    {
      for (int dev = 0; dev < num_devices; ++dev)
	{
	  int next = (dev + 1) % num_devices;
	  int to_next, from_next;
	  CHECK (hipDeviceCanAccessPeer (&to_next, dev, next));
	  CHECK (hipDeviceCanAccessPeer (&from_next, next, dev));
	  can_access[dev] = to_next && from_next;
	}
    }

  volatile control *ctl;
  CHECK (hipHostMalloc ((void **) &ctl, sizeof (control),
			hipHostMallocCoherent));

  /* Break after device query.  */
  int ret = 0;
  for (int dev = 0; dev < num_devices; ++dev)
    {
      if (!can_access[dev])
	continue;

      int dst_dev = (dev + 1) % num_devices;
      int base = (dev + 1) * 100;
      src_dev = dev;

      int host[COUNT];
      for (int i = 0; i < COUNT; ++i)
	host[i] = base + i;

      ctl->hold = 1;
      ctl->src_started = 0;
      ctl->dst_started = 0;

      /* Four ints wide, four rows, four slices.  */
      hipExtent extent = make_hipExtent (4 * sizeof (int), 4, 4);

      enable_peer (src_dev, dst_dev);
      int *src_2d;
      size_t src_pitch_2d;
      hipPitchedPtr src_3d;
      CHECK (hipMalloc ((int **) &src, sizeof (host)));
      CHECK (hipMemcpy (src, host, sizeof (host), hipMemcpyHostToDevice));
      CHECK (hipMallocPitch ((void **) &src_2d, &src_pitch_2d,
			     COLS * sizeof (int), ROWS));
      CHECK (hipMemcpy2D (src_2d, src_pitch_2d, host, COLS * sizeof (int),
			  COLS * sizeof (int), ROWS, hipMemcpyHostToDevice));
      CHECK (hipMalloc3D (&src_3d, extent));
      CHECK (hipMemset3D (src_3d, dev + 1, extent));
      CHECK (hipDeviceSynchronize ());

      /* The waves that stay running use non-blocking streams.  Otherwise
	 the copies and bump_kernel on the null stream would wait for them
	 and never run.  */
      hipStream_t src_stream;
      CHECK (hipStreamCreateWithFlags (&src_stream, hipStreamNonBlocking));
      src_kernel<<<1, 1, 0, src_stream>>> (&ctl->hold, &ctl->src_started);

      enable_peer (dst_dev, src_dev);
      int *dst_2d;
      size_t pitch_2d;
      hipPitchedPtr dst_3d;
      CHECK (hipMalloc ((int **) &dst, sizeof (host)));
      CHECK (hipMemset (dst, 0, sizeof (host)));
      CHECK (hipMalloc ((int **) &dst_async, sizeof (host)));
      CHECK (hipMemset (dst_async, 0, sizeof (host)));
      CHECK (hipMallocPitch ((void **) &dst_2d, &pitch_2d,
			     COLS * sizeof (int), ROWS));
      dst_2d_row = (int *) ((char *) dst_2d + pitch_2d);
      CHECK (hipMemset2D (dst_2d, pitch_2d, 0, COLS * sizeof (int), ROWS));
      CHECK (hipMalloc3D (&dst_3d, extent));
      CHECK (hipMemset3D (dst_3d, 0, extent));
      dst_3d_elem = (int *) ((char *) dst_3d.ptr
			     + (dst_3d.ysize + 1) * dst_3d.pitch);

      /* Make sure the destination device's queue is mapped before the
	 test inserts watchpoints.  */
      warmup_kernel<<<1, 1>>> ();
      CHECK (hipDeviceSynchronize ());

      hipStream_t dst_stream;
      CHECK (hipStreamCreateWithFlags (&dst_stream, hipStreamNonBlocking));
      dst_kernel<<<1, 1, 0, dst_stream>>> (&ctl->hold, &ctl->dst_started);
      while (!ctl->src_started || !ctl->dst_started)
	;

      /* Break before sync copy.  Device-to-device copies can return
	 before the data arrives, so wait for the destination device's
	 null stream after each one.  */
      CHECK (hipMemcpyPeer (dst, dst_dev, src, src_dev, sizeof (host)));
      CHECK (hipStreamSynchronize (nullptr));

      /* Break after sync copy.  */
      ctl->gate = 1;
      ctl->gate_started = 0;
      hipStream_t copy_stream;
      CHECK (hipStreamCreateWithFlags (&copy_stream, hipStreamNonBlocking));
      gate_kernel<<<1, 1, 0, copy_stream>>> (&ctl->gate, &ctl->gate_started);
      while (!ctl->gate_started)
	;
      CHECK (hipMemcpyPeerAsync (dst_async, dst_dev, src, src_dev,
				 sizeof (host), copy_stream));
      query_in_flight = hipStreamQuery (copy_stream);

      /* Break while async copy in flight.  */
      ctl->gate = 0;
      CHECK (hipStreamSynchronize (copy_stream));
      query_done = hipStreamQuery (copy_stream);

      /* Break after async copy.  */
      bump_kernel<<<1, 1>>> (dst_async);
      CHECK (hipStreamSynchronize (nullptr));

      CHECK (hipMemcpy2D (dst_2d, pitch_2d, src_2d, src_pitch_2d,
			  COLS * sizeof (int), ROWS, hipMemcpyDeviceToDevice));
      CHECK (hipStreamSynchronize (nullptr));

      /* Break after 2D copy.  */
      hipMemcpy3DPeerParms params = {};
      params.srcPtr = src_3d;
      params.srcDevice = src_dev;
      params.dstPtr = dst_3d;
      params.dstDevice = dst_dev;
      params.extent = extent;
      CHECK (hipMemcpy3DPeer (&params));
      CHECK (hipStreamSynchronize (nullptr));

      /* Break after 3D copy.  */
      ctl->hold = 0;
      CHECK (hipStreamSynchronize (dst_stream));
      CHECK (hipStreamDestroy (dst_stream));
      CHECK (hipStreamDestroy (copy_stream));

      int first, async_first;
      CHECK (hipMemcpy (&first, dst, sizeof (int), hipMemcpyDeviceToHost));
      CHECK (hipMemcpy (&async_first, dst_async, sizeof (int),
			hipMemcpyDeviceToHost));
      CHECK (hipFree (dst));
      CHECK (hipFree (dst_async));
      CHECK (hipFree (dst_2d));
      CHECK (hipFree (dst_3d.ptr));

      CHECK (hipSetDevice (src_dev));
      CHECK (hipStreamSynchronize (src_stream));
      CHECK (hipStreamDestroy (src_stream));
      CHECK (hipFree (src));
      CHECK (hipFree (src_2d));
      CHECK (hipFree (src_3d.ptr));

      if (first != base || async_first != base + BUMP)
	{
	  fprintf (stderr, "unexpected values %d, %d copying GPU %d to %d\n",
		   first, async_first, src_dev, dst_dev);
	  ret = 1;
	}
    }

  CHECK (hipHostFree ((void *) ctl));
  delete[] pci_location;
  delete[] can_access;
  return ret;
}
