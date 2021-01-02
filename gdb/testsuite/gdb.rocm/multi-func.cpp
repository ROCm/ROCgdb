/* Copyright (C) 2019-2021 Free Software Foundation, Inc.
   Copyright (C) 2019-2021 Advanced Micro Devices, Inc. All rights reserved.

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
#include <iostream>
#include "hip/hip_runtime.h"

#define CHECK(cmd)                                                                                 \
    {                                                                                              \
        hipError_t error = cmd;                                                                    \
        if (error != hipSuccess) {                                                                 \
            fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(error), error,         \
                    __FILE__, __LINE__);                                                           \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    }

// Defining number of elements in Array
#define N 5

__device__ int add_one(int no) {
    return no + 1;
}

__device__ int multiply_two(int no) {
    no = no*2;
    return no;
}

__device__ int multiply_three(int no) {
    no = no*3;
    return no;
}

//c = a*2 + b*3 + 1;
__device__ int gpu_operations(int a,int b) {
    int c;
    c = multiply_two(a) + multiply_three(b);
    c = add_one(c);
    return c;
}

__global__ void gpu_kernel_operations(int *d_a, int *d_b, int *d_c) {
    // Getting block index of current kernel
    int tid = blockIdx.x; // handle the data at this index
    if (tid < N)
        d_c[tid] = gpu_operations(d_a[tid],d_b[tid]);
}

int main(void) {
    // Defining host arrays
    int h_a[N], h_b[N], h_c[N];
    // Defining device pointers
    int *d_a, *d_b, *d_c;
    // allocate the memory
    CHECK(hipMalloc((void**)&d_a, N * sizeof(int)));
    CHECK(hipMalloc((void**)&d_b, N * sizeof(int)));
    CHECK(hipMalloc((void**)&d_c, N * sizeof(int)));

    // Initializing Arrays
    for (int i = 0; i < N; i++) {
        h_a[i] = i;
        h_b[i] = i + 1;
    }

    // Copy input arrays from host to device memory
    CHECK(hipMemcpy(d_a, h_a, N * sizeof(int), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(d_b, h_b, N * sizeof(int), hipMemcpyHostToDevice));

    // Calling kernels with N blocks and one thread per block, passing
    // device pointers as parameters
    hipLaunchKernelGGL(gpu_kernel_operations, dim3(N), dim3(1 ), 0, 0, d_a, d_b, d_c);

    // Copy result back to host memory from device memory
    CHECK(hipMemcpy(h_c, d_c, N * sizeof(int), hipMemcpyDeviceToHost));

    // Printing result on console
    for (int i = 0; i < N; i++) {
        if (h_c[i] !=  h_a[i]*2 + h_b[i]*3 + 1) {
            fprintf(stderr, "ERROR: wrong result of %d element is %d*2 + %d*3 + 1 = %d\n",
                   i, h_a[i], h_b[i],h_c[i]);
            exit(EXIT_FAILURE);
        }
    }
    // Free up memory
    CHECK(hipFree(d_a));
    CHECK(hipFree(d_b));
    CHECK(hipFree(d_c));
    return 0;
}

