/* Copyright (C) 2022 Free Software Foundation, Inc.
   Copyright (C) 2022 Advanced Micro Devices, Inc. All rights reserved.

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
#include <iostream>
#include <vector>
#include <fstream>

#define CHECK(cmd)                                                           \
  {                                                                          \
    hipError_t error = cmd;                                                  \
    if (error != hipSuccess) {                                               \
	fprintf(stderr, "error: '%s'(%d) at %s:%d\n",                        \
		hipGetErrorString(error), error, __FILE__, __LINE__);        \
	  exit(EXIT_FAILURE);                                                \
    }                                                                        \
  }

__device__ void
printId ()
{
  printf ("Hello from module (%d, %d, %d)\n",
	  (int) threadIdx.x, (int) threadIdx.y, (int) threadIdx.z);
}

extern "C" __global__ void
kernel ()
{
  printId ();
}

int
main (int argc, char* argv[])
{
  if (argc != 2)
    {
      std::cerr << "Usage: " << argv[0] << " module.hipbfb\n";
      return -1;
    }

  /* Copy the content of the module file into a memory buffer.  */
  const char *modpath = argv[1];
  std::ifstream mod (modpath, std::ios::binary | std::ios::ate);
  size_t module_size = mod.tellg ();
  mod.seekg (0, std::ios::beg);
  std::vector<char> module_buffer (module_size);
  if (!mod.read (module_buffer.data (), module_size))
    {
      std::cerr << "Failed to load HIP module into memory\n";
      return -1;
    }
  mod.close ();

  hipModule_t m;
  CHECK (hipModuleLoadData (&m, module_buffer.data ()));

  hipFunction_t f;
  CHECK (hipModuleGetFunction (&f, m, "kernel"));

  CHECK (hipModuleLaunchKernel (f, 1, 1, 1, 1, 1, 1, 0, 0, nullptr, nullptr));
  /* Now that the module is submitted to the device, try to be the worst
     possible citizen by unloading the module and scrambling the underlying
     buffer.  */
  hipModuleUnload (m);
  std::fill (module_buffer.begin (), module_buffer.end (), 0);
  module_buffer.resize (0);
  module_buffer.shrink_to_fit ();

  hipDeviceSynchronize ();
}
