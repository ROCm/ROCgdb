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

/* Build a HIP graph by capturing two kernel launches from a stream into a
   single graph, instantiate it, and then launch (replay) the executable
   graph several times with hipGraphLaunch.  This exercises the debugger's
   ability to associate stops with the right kernel and source line when the
   kernels reach the GPU through a graph launch -- the host submits a whole
   pre-recorded graph of operations with hipGraphLaunch, which dispatches the
   kernels -- rather than through a direct kernel<<<>>>() call.

   The two kernels operate on the same buffer, in order, so each graph
   replay computes out = (out + 1) * 3.  Starting from 0 this gives 3, 12
   and 39 after the first, second and third replay respectively.  The
   deterministic values let the .exp verify that each node executed in the
   right order with the right data on every replay.  */

#include <stdio.h>
#include <hip/hip_runtime.h>

#include "rocm-test-utils.h"

/* Number of times the executable graph is replayed.  The .exp overrides
   this via -DNUM_REPLAYS=... so the source and test stay in sync; the
   default lets the program build on its own.  */
#ifndef NUM_REPLAYS
#define NUM_REPLAYS 3
#endif

/* First graph node.  */

__global__ void
add_one (int *out)
{
  int tid = threadIdx.x;
  out[tid] = out[tid] + 1; /* break add_one */
}

/* Second graph node, run after add_one on the same buffer.  */

__global__ void
times_three (int *out)
{
  int tid = threadIdx.x;
  out[tid] = out[tid] * 3; /* break times_three */
}

int
main ()
{
  constexpr unsigned int num_elems = 1;

  int *result_ptr;
  CHECK (hipMalloc (&result_ptr, num_elems * sizeof (int)));
  CHECK (hipMemset (result_ptr, 0, num_elems * sizeof (int)));

  hipStream_t stream;
  CHECK (hipStreamCreate (&stream));

  /* Capture two kernel launches into a single graph.  */
  CHECK (hipStreamBeginCapture (stream, hipStreamCaptureModeGlobal));
  add_one<<<dim3 (1), dim3 (num_elems), 0, stream>>> (result_ptr);
  times_three<<<dim3 (1), dim3 (num_elems), 0, stream>>> (result_ptr);
  hipGraph_t graph;
  CHECK (hipStreamEndCapture (stream, &graph));

  /* Turn the graph into an executable graph.  */
  hipGraphExec_t graph_exec;
  CHECK (hipGraphInstantiate (&graph_exec, graph, nullptr, nullptr, 0));

  /* This is the "launch graph" execution path: instead of the host
     issuing each kernel<<<>>>(), hipGraphLaunch submits the whole graph and
     re-dispatches both kernels on every replay.  The debugger should stop
     in them on each launch.  */
  for (int i = 0; i < NUM_REPLAYS; i++)
    {
      CHECK (hipGraphLaunch (graph_exec, stream));
      CHECK (hipStreamSynchronize (stream));
    }

  int result;
  CHECK (hipMemcpyDtoH (&result, result_ptr, sizeof (int)));
  printf ("result is %d\n", result);

  CHECK (hipGraphExecDestroy (graph_exec));
  CHECK (hipGraphDestroy (graph));
  CHECK (hipStreamDestroy (stream));
  CHECK (hipFree (result_ptr));

  return 0;
}
