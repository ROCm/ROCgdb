/* This testcase is part of GDB, the GNU debugger.

   Copyright (C) 2026 Free Software Foundation, Inc.
   Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

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

#include <iostream>
#include <cassert>
#include <hip/hip_runtime.h>
#include "rocm-test-utils.h"

/* If set, the "hip_* ()" functions will record the error parameters.  */
static bool gen_ref = false;

struct hiperr_params_ref
{
  int no;
  const char *name;
  const char *str;

  void save (hipError_t err_no)
  {
    no = static_cast <int> (err_no);
    name = hipGetErrorName (err_no);
    str = hipGetErrorString (err_no);
  }
};

/* Reference values for the test.  */
static hiperr_params_ref hip_set_device_err;
static hiperr_params_ref hip_get_device_err;
static hiperr_params_ref hip_launch_kernel_err;

/* Get the maximum number of threads per block.  */

static size_t
get_max_block ()
{
  static size_t max_block = 0;

  if (max_block == 0)
    {
      int deviceId;
      CHECK (hipGetDevice (&deviceId));
      hipDeviceProp_t props;
      CHECK (hipGetDeviceProperties (&props, deviceId));
      max_block = props.maxThreadsPerBlock;
    }

  return max_block;
}

/* An empty kernel used for an oversized dispatch.  */

__global__ void
kernel ()
{
}

/* The purpose of these "hip_* ()" functions is to produce some errors
   and have the error parameters recorded.  GDB can use these recorded
   values as references to verify the catchpoints outputs.  */

/* Dispatch a kernel via the triple chevron syntax with an oversized
   dimension, which in turn launches the kernel using HIP APIs.  */

static void
hip_launch_kernel ()
{
  size_t max_block = get_max_block ();
  assert (max_block != 0);
  kernel<<<1, max_block + 1>>> ();
  hipError_t err = hipGetLastError ();
  assert (hipErrorInvalidConfiguration == err);

  if (gen_ref)
    hip_launch_kernel_err.save (err);
}

/* Set a device with bad ID.  */

static void
hip_set_device ()
{
  constexpr int bogus = 87654321;
  hipError_t err = hipSetDevice (bogus);
  assert (hipErrorInvalidDevice == err);

  if (gen_ref)
    hip_set_device_err.save (err);
}

/* Get a device with nullptr.  */

static void
hip_get_device ()
{
  hipError_t err = hipGetDevice (nullptr);
  assert (hipErrorInvalidValue == err);

  if (gen_ref)
    hip_get_device_err.save (err);
}

int
main (int argc, const char **argv)
{
  if (argc != 2)
    {
      std::cerr << "Usage: " << argv[0] << " ref|one|two|launch" << std::endl;
      return EXIT_FAILURE;
    }

  const std::string test (argv[1]);
  if (test == "ref")
    {
      gen_ref = true;
      hip_set_device ();
      hip_get_device ();
      hip_launch_kernel ();
      /* Break after reference initialization.  */
    }
  else if (test == "one")
    hip_set_device ();
  else if (test == "two")
    {
      hip_set_device ();
      hip_get_device ();
    }
  else if (test == "launch")
    hip_launch_kernel ();
  else
    {
      std::cerr << "Unsupported test " << test << std::endl;
      return EXIT_FAILURE;
    }

  return EXIT_SUCCESS;
}
