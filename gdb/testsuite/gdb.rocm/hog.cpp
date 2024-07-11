/* Copyright (C) 2021-2024 Advanced Micro Devices, Inc. All rights reserved.

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
#include <sstream>
#include <iomanip>
#include <fstream>
#include <dirent.h>
#include <iostream>
#include <memory>
#include <optional>

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

struct dispatch_size
{
  dim3 blocks;
  dim3 threadsPerBlock;
};

/* Compute the dispatch size necessary to occupy most but not all of the
   resources of device HIP_DEVICE_ID.

   We aim at occupying half of the GPU wave slots.

   If the exact topology of the GPU cannot be accessed, return a default
   dispatch size which might not be accurate.  */

static dispatch_size
hog_dispatch_size (int hip_device_id)
{
  hipDeviceProp_t props;
  CHECK (hipGetDeviceProperties (&props, hip_device_id));

  static const std::string kfd_topo_nodes_path
    { "/sys/devices/virtual/kfd/kfd/topology/nodes/" };
  std::unique_ptr<DIR, void (*) (DIR *)> dirp
    { opendir (kfd_topo_nodes_path.c_str ()),
      [] (DIR *d) { closedir (d); } };

  if (dirp == nullptr)
    {
      perror ("Cannot access KFD topology");
      exit (EXIT_FAILURE);
    }

  struct dirent *dir;
  while ((dir = readdir (dirp.get ())) != nullptr)
    {
      if (strcmp (dir->d_name, ".") == 0
	  || strcmp (dir->d_name, "..") == 0)
	continue;

      std::ifstream fs { kfd_topo_nodes_path + dir->d_name + "/properties" };

      if (!fs.good ())
	continue;

      std::optional<uint32_t> kfd_max_waves_per_simd;
      std::optional<uint32_t> kfd_simd_count;
      std::optional<uint32_t> kfd_pci_domain;
      std::optional<uint32_t> kfd_pci_location;

      std::string name;
      uint32_t value;
      while (fs >> name >> value)
	{
	  if (name == "max_waves_per_simd")
	    kfd_max_waves_per_simd.emplace (value);
	  else if (name == "simd_count")
	    kfd_simd_count.emplace (value);
	  else if (name == "domain")
	    kfd_pci_domain.emplace (value);
	  else if (name == "location_id")
	    kfd_pci_location.emplace (value);
	}

      if (kfd_max_waves_per_simd.has_value ()
	  && kfd_simd_count.has_value ()
	  && kfd_pci_domain.has_value ()
	  && kfd_pci_location.has_value ()
	  && props.pciDomainID == kfd_pci_domain.value ()
	  && props.pciBusID == ((kfd_pci_location.value () >> 8) & 0xff)
	  && props.pciDeviceID == ((kfd_pci_location.value () >> 3) & 0x1f))
	{
	  return {{kfd_simd_count.value ()
		   * (kfd_max_waves_per_simd.value () / 2)},
		{static_cast<uint32_t> (props.warpSize)}};
	}
    }

  std::cerr << "Cannot find device topology.  Using arbitrary dispatch size."
	    << std::endl;
  return {{512}, {256}};
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

  auto dim = hog_dispatch_size (deviceId);

  hipLaunchKernelGGL (hog_kernel, dim.blocks, dim.threadsPerBlock, 0, 0);
  CHECK (hipDeviceSynchronize ());
  return 0;
}

