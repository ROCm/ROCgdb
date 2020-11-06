/* Copyright (C) 2019-2020 Free Software Foundation, Inc.
   Copyright (C) 2019-2020 Advanced Micro Devices, Inc. All rights reserved.

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

#include "hip/hip_runtime.h"
#include "stdio.h"
#include <iostream>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime.h>

// Number of elements in Array.
#define N 64

#define HIPCHECK(cmd)                                                          \
do {                                                                           \
  hipError_t error = (cmd);                                                    \
  if (error != hipSuccess)                                                     \
  {                                                                            \
    std::cerr << "Encountered HIP error (" << error << ") at line "            \
              << __LINE__ << " in file " << __FILE__ << "\n";                  \
    exit(-1);                                                                  \
  }                                                                            \
} while (0)

#define MAX_GPU 8



// Defining Kernel function for vector addition
__global__ void VectorAdd(int *d_a, int *d_b, int *d_c)
{
  // Getting block index of current kernel
  int tid = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
  if (tid < N)
    d_c[tid] = d_a[tid] + d_b[tid];
}


int main(void)
{
  // Defining host arrays
  int h_a[N], h_b[N], h_c[N];
  // Defining device pointers
  int *d_a[N], *d_b[N], *d_c[N];
  // allocate the memory
  
  hipStream_t stream[MAX_GPU];

  int nGpu = 1;
  // To do
  // In multi gpu scenario on running kernel on every GPU causing multiple
  // failures in nonstop mode test cases, which need to investigate.
  // HIPCHECK(hipGetDeviceCount(&nGpu));
  for (int i = 0; i < nGpu; i ++) {
    HIPCHECK(hipSetDevice(i));
    hipDeviceProp_t prop;
    HIPCHECK(hipGetDeviceProperties(&prop, i));
    printf("#   device %d [0x%02x] %s\n",
                    i, prop.pciBusID, prop.name);
    //create stream
    HIPCHECK(hipStreamCreate(&stream[i]));

    hipMalloc((void**)&d_a[i], N * sizeof(int));
    hipMalloc((void**)&d_b[i], N * sizeof(int));
    hipMalloc((void**)&d_c[i], N * sizeof(int));
    // Initializing Arrays
    for (int i = 0; i < N; i++) {
      h_a[i] = 2*i;
      h_b[i] = i ;
    }

    // Copy input arrays from host to device memory
    hipMemcpyAsync(d_a[i], h_a, N * sizeof(int), hipMemcpyHostToDevice, stream[i]);
    hipMemcpyAsync(d_b[i], h_b, N * sizeof(int), hipMemcpyHostToDevice, stream[i]);
  }
  
  for (int i = 0; i < nGpu; i ++) {
  HIPCHECK(hipSetDevice(i));

  hipLaunchKernelGGL(VectorAdd,
		     dim3(GRID_DIM), dim3(BLOCK_DIM),
		     0, stream[i], d_a[i], d_b[i], d_c[i]);
  }
  
  for (int i = 0; i < nGpu; i ++) {
    HIPCHECK(hipSetDevice(i));
    // Copy result back to host memory from device memory
    hipMemcpyAsync(h_c, d_c[i], N * sizeof(int), hipMemcpyDeviceToHost, stream[i]);
    HIPCHECK(hipStreamSynchronize(stream[i]));
    //printf("Vector addition on GPU \n");
    // Printing result on console
    for (int i = 0; i < N; i++) {
      /*printf("Operation result of %d element is %d + %d = %d\n",
         i, h_a[i], h_b[i],h_c[i]);*/
      if(h_a[i]+h_b[i] !=h_c[i]) {
        HIPCHECK(hipErrorUnknown); 
      }
    }
    // Free up memory
    HIPCHECK(hipStreamDestroy(stream[i]));
    hipFree(d_a[i]);
    hipFree(d_b[i]);
    hipFree(d_c[i]);
  }
  return 0;
}
