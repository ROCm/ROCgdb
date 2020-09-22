/* Wrapper translation unit for testing against AMDGCN devices, using HIP.

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

/* All compilations are wrapped with this file.  INC_TEST_FILE must be
   defined to the actual file we want to compile.  */

#include <hip/hip_runtime.h>

/* The hip compiler defines this as a built-in, which conflicts with
   testcases defining it themselves.  */
#undef _GNU_SOURCE

/* Avoid having to write explicit __device__ in all functions
   throughout.  */
#pragma clang force_cuda_host_device begin

/* Same for global variables.  */

/* Need this otherwise Clang warns when the __device__ attribute isn't
   applied to anything.  */
#pragma GCC diagnostic ignored "-Wpragma-clang-attribute"

#pragma clang attribute gdb_device_globals.push (__device__, apply_to = variable (is_global))

/* Pull in the gdb_hip_test_main declarations.  */
#include "hip-test-main.h"

/* Rewire the test's "main" to the function that the kernel calls.
   The real "main" is the host's main.  */
#define main gdb_hip_test_main

/* Include the actual testcase source file.  */
#include INC_TEST_FILE

/* Pop the scope.  Unlike with all other push/pop or begin/end
   pragma's, Clang errors out if you don't do this with #pca.  This is
   the reason we use a wrapper translation unit instead of a header
   pulled in via -include on the command line.  */
#pragma clang attribute gdb_device_globals.pop
