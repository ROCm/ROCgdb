/* This testcase is part of GDB, the GNU debugger.

   Copyright 2024 Free Software Foundation, Inc.
   Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.

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

#define CHECK(cmd)								\
  {										\
    hipError_t error = cmd;							\
    if (error != hipSuccess)							\
      {										\
	fprintf (stderr, "error: '%s'(%d) at %s:%d\n",				\
		 hipGetErrorString (error), error, __FILE__, __LINE__);		\
	exit (EXIT_FAILURE);							\
      }										\
  }

__global__ void
use_shared (size_t shared_size_bytes)
{
  constexpr size_t shared_buffer_size = SHARED_SIZE / 4;
  __shared__ uint32_t shared_buffer[shared_buffer_size];

  /* Helper function generating the value to store in share memory.  IDX is
     the position in the shared buffer.

     This function is called once to generate values to store in shared memory,
     and a second time to validate values read from shared.  Between those two
     calls, it is expected that the debugger has updated the memory for one of
     the work-groups in the dispatch.  The UPDATED_BY_DEBUGGER flag indicates
     that we should generate the values by taking into account the updates the
     debugger should have made.

     The generated values contain:
     - "marker" values at the beginning, middle and end of the buffer,
     - data that can vary from one work-group to the next (we use the
       work-group coordinates for that),
     - known padding data everywhere else.
  */
  auto generator
    = [shared_buffer_size] (size_t idx, bool updated_by_debugger) -> uint32_t
    {
      /* Beginning marker of the buffer marker.  */
      if (idx == 0)
	return updated_by_debugger ? 0 : 0xdeadbeef;

      /* Data that can vary from one work-group to the next.  */
      if (idx == 1)
	return blockIdx.x;
      if (idx == 2)
	return blockIdx.y;
      if (idx == 3)
	return blockIdx.z;

      /* Middle of the buffer marker.  */
      if (idx == shared_buffer_size / 2)
	return updated_by_debugger ? 0xffffffff : 0xbadf00d;

      /* End of the buffer marker.  */
      if (idx == shared_buffer_size - 1)
	return updated_by_debugger ? 0xabcd0123 : 0x0123abcd;

      /* Anything else.  */
      return 0xaaaaaaaa;
    };

  /* Initialize the shared buffer with known values.  */
  for (size_t idx = threadIdx.x; idx < shared_buffer_size; idx += blockDim.x)
    shared_buffer[idx] = generator (idx, false);

  /* Insert a breakpoint after all waves are done initializing shared
     memory.  */
  __syncthreads ();

  /* Break here.  */

  /* The debugger only stops one wave in the selected work-group.  Make sure
     that all waves in the work-group synchronize here to avoid waves not
     stopped by the debugger starting to validate the content of shared_buffer
     before the debugger has finished updating it.  */
  __syncthreads ();

  /* Check that after the breakpoint has been hit, shared memory still
     contains the expected values.  */
  for (size_t idx = threadIdx.x; idx < shared_buffer_size; idx += blockDim.x)
    {
      /* We expect that the debugger has modified a couple of values in the
	 shared buffer, only for work-group with X == 23.  */
      const bool modified_by_debugger = blockIdx.x == 23;
      assert (shared_buffer[idx] == generator (idx, modified_by_debugger));
    }
}

int
main (int argc, char **argv)
{
  int deviceId;
  CHECK (hipGetDevice (&deviceId));

  hipDeviceProp_t props;
  CHECK (hipGetDeviceProperties(&props, deviceId));

  assert (props.sharedMemPerBlock == SHARED_SIZE);

  use_shared<<<128, props.maxThreadsPerBlock>>> (props.sharedMemPerBlock);

  CHECK (hipDeviceSynchronize ());
}
