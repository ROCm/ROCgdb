/* Wrapper for Clang's __clang_hip_runtime_wrapper.h.

   Copyright (C) 2021 Free Software Foundation, Inc.
   Copyright (C) 2021 Advanced Micro Devices, Inc. All rights reserved.

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

/* Clang automatically includes __clang_hip_runtime_wrapper.h via
   -include on the command line.  We need to override the malloc
   definition so we need to wrap the wrapper header...  We do that by
   having a file (this one) with the same name in the include path.
   We pull the original via #include_next.  */

/* The HIP-provided malloc is broken (see comments around
   __HIP_ENABLE_DEVICE_MALLOC__), so we provide our own very simple
   malloc/free.  */

/* __clang_hip_runtime_wrapper.h includes stdlib.h, so include it
   ourselves before malloc is renamed, so that we get the __host__
   version declared.  */
#include <cmath>
#include <cstdlib>
#include <stdlib.h>

#define malloc broken_hip_malloc
#define free broken_hip_free

#include_next <__clang_hip_runtime_wrapper.h>

#undef malloc
#undef free

/* /opt/rocm/hip/include/hip/hcc_detail/host_defines.h defines this,
   but it conflicts with uses such as:
   __attribute__((__noinline__))
*/
#undef __noinline__

/* stdlib.h only declares the __host__ versions.  Declare them for the
   device.  */
extern "C" __device__ void *malloc (size_t size)
  __attribute__ ((malloc));
extern "C" __device__ void *calloc (size_t nmemb, size_t size)
  __attribute__ ((malloc));
extern "C" __device__ void *free (void *ptr);

/* Declare some more standard functions used by tests that are missing
   on the device.  Defined in hip-driver.cc.  */

extern "C" __device__ int atoi (const char *nptr);

extern "C" __device__ void exit (int) __attribute__ ((noreturn));

extern "C" __device__ int puts (const char *s);

extern "C" __device__ size_t strlen (const char *s);

extern "C" __device__ char *strdup (const char *s);

extern "C" __device__ int strncmp (const char *s1, const char *s2, size_t n);

extern "C" __device__ char *strcpy (char *dest, const char *src);

extern "C" __device__ char *getenv (const char *name);
