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

#define CHECK(cmd)                                                           \
    {                                                                        \
	hipError_t error = cmd;                                              \
	if (error != hipSuccess)                                             \
	  {                                                                  \
	    fprintf(stderr, "error: '%s'(%d) at %s:%d\n",                    \
		    hipGetErrorString(error), error,                         \
		    __FILE__, __LINE__);                                     \
	    exit(EXIT_FAILURE);                                              \
	  }                                                                  \
    }

__global__ void
hog_kernel ()
{
  while (true)
    __builtin_amdgcn_s_sleep (1);
}

int
main ()
{
  int deviceId;
  CHECK (hipGetDevice (&deviceId));

  hipDeviceProp_t props;
  CHECK (hipGetDeviceProperties (&props, deviceId));

  /* Heuristics to scale the `hog` kernel for occupying a significant portion
     of the GPU's concurrent wave execution capacity, regardless of GPU size.

     Target: ~432 total waves (works well on 36 CU GPU:
     at least 12 waves per CU: 36 x 12 = 432)
     - Small GPUs (<= 8 CUs): scale #_of_CUs by some factor to ensure saturation
       the factor is ad hoc 50 as 8 x 50 = 400 close to target
     - Medium/Large GPUs: Use 432 waves or 12×CU count, whichever is larger

     This ensures:
     1. Small GPUs get enough waves to create resource pressure
     2. Large GPUs don't launch excessively many waves
     3. Total wave count is predictable and reasonable
  */

  const size_t threads_per_wave = props.warpSize; // either 32 or 64
  constexpr size_t THREADS_PER_BLOCK = 256;
  constexpr size_t TINY_GPU_CU_COUNT = 8;
  constexpr size_t BIG_GPU_CU_COUNT = 36;
  constexpr size_t WAVES_PER_CU_4_TINY = 50;
  constexpr size_t MAX_WAVES_PER_CU = 12;
  constexpr size_t TARGET_WAVES = BIG_GPU_CU_COUNT * MAX_WAVES_PER_CU; // i.e. 432
  constexpr size_t MIN_TARGET_WAVES = 100;
  constexpr size_t MAX_TARGET_WAVES = 5000;

  /* Calculate target waves */
  size_t target_waves;
  const char *env_waves_total = getenv ("HOG_TOTAL_WAVES");
  size_t cu_count = props.multiProcessorCount;

  if (env_waves_total != nullptr)
    target_waves = atoi (env_waves_total);
  else
    {
      target_waves = (cu_count > TINY_GPU_CU_COUNT)
      ? std::max (TARGET_WAVES, cu_count * MAX_WAVES_PER_CU)
      : cu_count * WAVES_PER_CU_4_TINY;
    }

  target_waves = std::clamp (target_waves, MIN_TARGET_WAVES, MAX_TARGET_WAVES);

  size_t blocks = (target_waves * threads_per_wave + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  size_t waves_per_cu = target_waves / cu_count;

  // Diagnostic output
  fprintf(stderr, "Hog: GPU '%s' with %zu CUs, wave size %zu\n",
    props.name, cu_count, threads_per_wave);
  fprintf(stderr, " Launching %zu blocks x %zu threads = %zu waves "
    "(~%zu waves/CU)\n", blocks, THREADS_PER_BLOCK, target_waves, waves_per_cu);

  hog_kernel<<<blocks, THREADS_PER_BLOCK>>> ();

  CHECK (hipDeviceSynchronize ());
  return 0;
}
