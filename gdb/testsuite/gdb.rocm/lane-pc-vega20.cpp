/* Copyright (C) 2021-2022 Free Software Foundation, Inc.
   Copyright (C) 2021-2022 Advanced Micro Devices, Inc. All rights reserved.

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

#include <cstring>
#include <hip/hip_runtime.h>
#include <iostream>

#define CHECK(cmd)                                                            \
  {                                                                           \
    hipError_t error = cmd;                                                   \
    if (error != hipSuccess)                                                  \
      {                                                                       \
	std::cerr << "error: '" << hipGetErrorString (error) << "'(" << error \
		  << ") at " << __FILE__ << ":" << __LINE__ << std::endl;     \
	exit (EXIT_FAILURE);                                                  \
      }                                                                       \
  }

static constexpr size_t LANE_NUM = 32;
static constexpr char ARCH[] = "gfx906";

int
main (int argc, char *argv[])
{
  int device_id, device_count;

  CHECK (hipGetDeviceCount (&device_count));
  for (device_id = 0; device_id < device_count; ++device_id)
    {
      hipDeviceProp_t properties;
      CHECK (hipGetDeviceProperties (&properties, device_id));

      if (!strncmp (properties.gcnArchName, ARCH, sizeof (ARCH) - 1))
	{
	  std::cout << "Running on device #" << device_id << " "
		    << properties.name << std::endl;
	  CHECK (hipSetDevice (device_id));
	  break;
	}
    }

  if (device_id == device_count)
    {
      std::cerr << "Cannot find a suitable device to run the test."
		<< std::endl;
      exit (EXIT_FAILURE);
    }

  /* Load kernel binary.  KERNEL_SO_NAME is the name of the .so file,
     and is defined by the .exp file.  Try loading it from
     KERNEL_SO_PATH first (also defined by the .exp file), and
     fallback to loading it from the current directory.  The latter is
     useful for standalone development, particularly when developing
     non-GDB tools.  */
  hipModule_t module;
  if (hipModuleLoad (&module, KERNEL_SO_PATH "/" KERNEL_SO_NAME) != hipSuccess)
    CHECK (hipModuleLoad (&module, "./" KERNEL_SO_NAME));

  hipFunction_t function;
  CHECK (hipModuleGetFunction (&function, module, "AddrClassTest"));

  int in_h[LANE_NUM], *in_d;
  struct
  {
    int int_elem;
    char char_elem;
    int array_elem[LANE_NUM];
  } out_h, *out_d;

  CHECK (hipMalloc (&in_d, sizeof (in_h)));
  CHECK (hipMalloc (&out_d, sizeof (out_h)));

  for (size_t i = 0; i < LANE_NUM; ++i)
    in_h[i] = i;

  CHECK (hipMemcpy (in_d, &in_h, sizeof (in_h), hipMemcpyHostToDevice));

  struct
  {
    decltype (in_d) in;
    decltype (out_d) out;
  } params = { in_d, out_d };
  size_t params_size = sizeof (params);

  void *config[]
    = { HIP_LAUNCH_PARAM_BUFFER_POINTER, &params, HIP_LAUNCH_PARAM_BUFFER_SIZE,
	&params_size, HIP_LAUNCH_PARAM_END };

  CHECK (hipModuleLaunchKernel (function, 1, 1, 1, LANE_NUM, 1, 1, 0, 0,
				nullptr, (void **) &config));

  CHECK (hipMemcpy (&out_h, out_d, sizeof (out_h), hipMemcpyDeviceToHost));

  CHECK (hipModuleUnload (module));
  CHECK (hipFree (in_d));
  CHECK (hipFree (out_d));

  std::cout << "Results:" << std::endl;
  for (size_t i = 0; i < LANE_NUM; ++i)
    std::cout << "Lane " << i << ": " << out_h.array_elem[i] << std::endl;

  return 0;
}
