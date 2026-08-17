/* This testcase is part of GDB, the GNU debugger.

   Copyright 2026 Free Software Foundation, Inc.

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

#include "rocm-test-utils.h"
#include <hip/hip_runtime.h>
#include <fstream>
#include <vector>

/* Test file:// events (hipModuleLoad).  */
static void
test_file_load (const char *module_path)
{
  hipModule_t module;
  CHECK (hipModuleLoad (&module, module_path));

  hipFunction_t function;
  CHECK (hipModuleGetFunction (&function, module, "test_kernel"));

  CHECK (hipModuleLaunchKernel (function, 1, 1, 1, 1, 1, 1,
				0, nullptr, nullptr, nullptr));

  CHECK (hipDeviceSynchronize ());
  CHECK (hipModuleUnload (module));
}

/* Test memory:// events (hipModuleLoadData).  */
static void
test_memory_load (const char *module_path)
{
  /* Read module file into memory buffer.  */
  std::ifstream mod (module_path, std::ios::binary | std::ios::ate);
  if (!mod.is_open ())
    {
      fprintf (stderr, "Failed to open module file\n");
      exit (EXIT_FAILURE);
    }

  size_t module_size = mod.tellg ();
  mod.seekg (0, std::ios::beg);
  std::vector<char> module_buffer (module_size);

  if (!mod.read (module_buffer.data (), module_size))
    {
      fprintf (stderr, "Failed to read module into memory\n");
      exit (EXIT_FAILURE);
    }
  mod.close ();

  /* Load from memory buffer.  */
  hipModule_t module;
  CHECK (hipModuleLoadData (&module, module_buffer.data ()));

  hipFunction_t function;
  CHECK (hipModuleGetFunction (&function, module, "test_kernel"));

  CHECK (hipModuleLaunchKernel (function, 1, 1, 1, 1, 1, 1,
				0, nullptr, nullptr, nullptr));

  CHECK (hipDeviceSynchronize ());
  CHECK (hipModuleUnload (module));
}

int
main (int argc, char **argv)
{
  if (argc != 2)
    {
      fprintf (stderr, "Usage: %s <module_path>\n", argv[0]);
      return EXIT_FAILURE;
    }

  const char *module_path = argv[1];

  test_file_load (module_path);
  test_memory_load (module_path);

  return 0;
}
