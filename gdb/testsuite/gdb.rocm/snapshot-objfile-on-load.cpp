/* Copyright (C) 2022-2024 Free Software Foundation, Inc.
   Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.

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
#include <string_view>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <elf.h>

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

/* Extract an 8-byte integer from *BUF_P, and advance *BUF_P past
   it.  */

static uint64_t
extract_uint64 (char **buf_p)
{
  uint64_t var;
  memcpy (&var, *buf_p, sizeof (var));
  *buf_p += sizeof (var);
  return var;
}

int
main (int argc, char* argv[])
{
  if (argc != 3)
    {
      std::cerr << "Usage: " << argv[0]
		<< " zero|shoff"
		<< " module.hipbfb"
		<< std::endl;
      return EXIT_FAILURE;
    }

  std::string_view mode = argv[1];

  /* Copy the content of the module file into a memory buffer.  */
  const char *modpath = argv[2];
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
  CHECK (hipModuleUnload (m));

  if (mode == "zero")
    {
      std::cout << "zeroing module buffer\n";

      std::fill (module_buffer.begin (), module_buffer.end (), 0);
      module_buffer.resize (0);
      module_buffer.shrink_to_fit ();
    }
  else if (mode == "shoff")
    {
      std::cout << "corrupting section offsets in module buffer\n";

      /* Corrupt the section offset in the ELF header so that it ends
	 up pointing outside the code object.  GDB used to access
	 memory out of bounds (and often crash) in this scenario.  */

      /* Parse the bundled code object to find the ELFs (one per
	 architecture), as per:
	 https://clang.llvm.org/docs/ClangOffloadBundler.html#bundled-binary-file-layout.  */

      /* Skip magic string.  */
      char *p = module_buffer.data () + 24;

      uint64_t number_of_bundles = extract_uint64 (&p);
      std::cout << "number of bundles: " << number_of_bundles << std::endl;

      for (uint64_t i = 0; i < number_of_bundles; i++)
	{
	  uint64_t offset = extract_uint64 (&p);
	  uint64_t size = extract_uint64 (&p);
	  uint64_t ID_length = extract_uint64 (&p);
	  p += ID_length;

	  char *elf = &module_buffer[offset];

	  /* A large section offset that is out of bounds of code
	     object ELF, and thus out of the bounds of the internal
	     buffer GDB uses to hold a code object copy.  It is not
	     guaranteed that this would crash GDB, but it is
	     likely.  */
	  size_t new_offset = 0x7da26c862090;

	  if (elf[EI_CLASS] == ELFCLASS32)
	    {
	      Elf32_Off new_off32 = new_offset;
	      size_t e_shoff_offset = offsetof (Elf32_Ehdr, e_shoff);
	      memcpy (elf + e_shoff_offset, &new_offset, sizeof (new_off32));
	    }
	  else if (elf[EI_CLASS] == ELFCLASS64)
	    {
	      Elf64_Off new_off64 = new_offset;
	      size_t e_shoff_offset = offsetof (Elf64_Ehdr, e_shoff);
	      memcpy (elf + e_shoff_offset, &new_offset, sizeof (new_off64));
	    }
	  else
	    abort ();
	}
    }
  else
    {
      std::cerr << "unknown mode: " << mode << std::endl;
      return EXIT_FAILURE;
    }

  CHECK (hipDeviceSynchronize ());
}
