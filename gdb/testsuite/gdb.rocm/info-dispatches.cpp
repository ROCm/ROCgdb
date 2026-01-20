/* Copyright (C) 2025 Free Software Foundation, Inc.
   Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

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

#include <stdio.h>
#include <hip/hip_runtime.h>

#define CHECK(cmd)							  \
  do									  \
    {									  \
      hipError_t error = cmd;						  \
      if (error != hipSuccess)						  \
	{								  \
	  fprintf (stderr, "error: '%s'(%d) at %s:%d\n",		  \
		   hipGetErrorString (error), error, __FILE__, __LINE__); \
	  exit (EXIT_FAILURE);						  \
	}								  \
    }									  \
  while (0)

/* Make sure kernels running on STREAM are dispatched.  */

#define flush_dispatch(STREAM) \
  flush_dispatch_1 (STREAM, __FILE__, __LINE__)

static void
flush_dispatch_1 (hipStream_t stream, const char *file, int line)
{
  hipError_t error = hipStreamQuery (stream);
  if (error != hipSuccess && error != hipErrorNotReady)
    {
      fprintf (stderr, "error: '%s'(%d) at %s:%d\n",
	       hipGetErrorString (error), error, file, line);
      exit (EXIT_FAILURE);
    }
}

/* Two single-wave kernels to test `info dispatches` with
   multiple dispatches.  */

__global__ void
single_wave_kernel1 ()
{
}

__global__ void
single_wave_kernel2 ()
{
}

int
main (int argc, char* argv[])
{
  const unsigned int numBlocks = 1;
  const unsigned int numThreadsPerBlock = 1;
  const unsigned int sharedMemBytes = 0;
  hipStream_t stream1, stream2;

  CHECK (hipStreamCreate (&stream1));
  CHECK (hipStreamCreate (&stream2));

  printf ("info: launch 'single_wave_kernel1' and 'single_wave_kernel2'\n");
  single_wave_kernel1<<<numBlocks, numThreadsPerBlock, sharedMemBytes,
			stream1>>> ();
  single_wave_kernel2<<<numBlocks, numThreadsPerBlock, sharedMemBytes,
			stream2>>> ();

  /* Make sure both kernels are dispatched on Windows, otherwise they
     may only be launched by the hipStreamSynchronize calls below, and
     the first call would hang as kernel1 would be stopped at a
     breakpoint.  */
  flush_dispatch (stream1);
  flush_dispatch (stream2);

  CHECK (hipStreamSynchronize (stream1));
  CHECK (hipStreamSynchronize (stream2));

  CHECK (hipStreamDestroy (stream1));
  CHECK (hipStreamDestroy (stream2));

  return 0;
}
