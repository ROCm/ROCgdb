/* This testcase is part of GDB, the GNU debugger.

   Copyright 2022-2025 Free Software Foundation, Inc.

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

#if defined(SHARED_LIB_1)

extern "C"
{

  using shared_lib_1_t = int;

  void func_host_shared_lib_1 (shared_lib_1_t)
  {
  }

  void func_host_shared_lib (shared_lib_1_t)
  {
  }

  void func_1 (shared_lib_1_t)
  {
  }

  void func_main_shared_lib_1 (shared_lib_1_t)
  {
  }

} /* extern "C" */

#elif defined(SHARED_LIB_2)

extern "C"
{

  using shared_lib_2_t = int;

  void func_host_shared_lib_2 (shared_lib_2_t)
  {
  }

  void func_host_shared_lib (shared_lib_2_t)
  {
  }

  void func_2 (shared_lib_2_t)
  {
  }

  void func_main_shared_lib_2 (shared_lib_2_t)
  {
  }

} /* extern "C" */

#elif defined(CODE_OBJECT_1)

extern "C"
{

  using code_object_1_t = int;

  __global__ void func_device_code_object_1 (code_object_1_t)
  {
  }

  __global__ void func_device_code_object (code_object_1_t)
  {
  }

  __global__ void func_1 (code_object_1_t)
  {
  }

  __global__ void func_main_code_object_1 (code_object_1_t)
  {
  }

} /* extern "C" */

#elif defined(CODE_OBJECT_2)

extern "C"
{

  using code_object_2_t = int;

  __global__ void func_device_code_object_2 (code_object_2_t)
  {
  }

  __global__ void func_device_code_object (code_object_2_t)
  {
  }

  __global__ void func_2 (code_object_2_t)
  {
  }

  __global__ void func_main_code_object_2 (code_object_2_t)
  {
  }

} /* extern "C" */

#else

#include <dlfcn.h>
#include <iostream>
#include <memory>
#include <stdexcept>
#include "hip/hip_runtime.h"

/* This `extern "C"` is necessary to make the test work, due to an oddity in how
   GDB computes the demangled names.  Without it, the compiler includes a
   DW_AT_linkage_name in the DW_TAG_subprogram DIE, causing GDB to use that to
   compute the symbol's demangled name.  The symbols would then appear as
   `func_main(int)` rather than `func_main(main_program_t)`, making it
   impossible to differentiate them.  With `extern "C"`, GDB has to construct
   the demangled name from the DIE structure, resulting in the expected
   `func_main(main_program_t)`.   */

extern "C"
{

  using main_program_t = int;

  void func_main (main_program_t)
  {
  }

  void func_main_shared_lib_1 (main_program_t)
  {
  }

  void func_main_shared_lib_2 (main_program_t)
  {
  }

  void func_main_code_object_1 (main_program_t)
  {
  }

  void func_main_code_object_2 (main_program_t)
  {
  }

} /* extern "C" */

namespace
{

void
throw_from_hip_error (const std::string &prefix, hipError_t error)
{
  throw std::runtime_error (prefix + ": " + hipGetErrorString (error));
}

void
warn_from_hip_error (const std::string &prefix, hipError_t error)
{
  std::cerr << prefix + ": " + hipGetErrorString (error) << std::endl;
}

struct hip_module_unloader
{
  void operator() (hipModule_t module) const
  {
    hipError_t error = hipModuleUnload (module);

    if (error != HIP_SUCCESS)
      warn_from_hip_error ("Failed to unload HIP module", error);
  }
};

using hip_module_up
  = std::unique_ptr<std::remove_pointer_t<hipModule_t>, hip_module_unloader>;

hip_module_up
load_hip_module (const char *module_path)
{
  hipModule_t module;
  hipError_t error = hipModuleLoad (&module, module_path);

  if (error != HIP_SUCCESS)
    throw_from_hip_error ((std::string ("Failed to load HIP module ")
			   + module_path),
			  error);

  return hip_module_up (module);
}

void
throw_from_dlerror (const std::string &prefix)
{
  throw std::runtime_error (prefix + ": " + dlerror ());
}

void
warn_from_dlerror (const std::string &prefix)
{
  std::cerr << prefix + ": " + dlerror () << std::endl;
}

struct dlcloser
{
  void operator() (void *lib)
  {
    int ret = dlclose (lib);

    if (ret != 0)
      warn_from_dlerror ("Failed to dlclose");
  }
};

using dl_up = std::unique_ptr<void, dlcloser>;

dl_up
load_shared_lib (const char *lib_path)
{
  void *lib = dlopen (lib_path, RTLD_NOW);

  if (lib == nullptr)
    throw_from_dlerror (std::string ("Failed to dlopen ") + lib_path);

  return dl_up (lib);
}

using host_function_t = void (*) (int);

host_function_t
lookup_symbol (dl_up &dl, const char *sym_name)
{
  void *sym = dlsym (dl.get (), sym_name);

  if (sym == nullptr)
    throw_from_dlerror (std::string ("Failed to dlsym ") + sym_name);

  return reinterpret_cast<host_function_t> (sym);
}

hipFunction_t
lookup_symbol (hip_module_up &mod, const char *sym_name)
{
  hipFunction_t func;
  hipError_t error = hipModuleGetFunction (&func, mod.get (), sym_name);

  if (error != HIP_SUCCESS)
    throw_from_hip_error (std::string ("Failed to look up kernel ") + sym_name,
			  error);

  return func;
}

void
launch_kernel (hipFunction_t func)
{
  int arg = 2;
  void *args[1] = { &arg };

  hipError_t error = hipModuleLaunchKernel (func, 1, 1, 1, 1, 1, 1, 0, nullptr,
					    args, nullptr);

  if (error != HIP_SUCCESS)
    throw_from_hip_error ("Failed to launch kernel", error);

  error = hipDeviceSynchronize ();

  if (error != HIP_SUCCESS)
    throw_from_hip_error ("Failed to sync", error);
}

} /* namespace */

int
main (int argc, const char **argv)
{
  if (argc != 5)
    {
      fprintf (stderr, "Usage: %s <shlib1> <shlib2> <hip1> <hip1>\n", argv[0]);
      return 1;
    }

  dl_up shared_lib_1 = load_shared_lib (argv[1]);
  dl_up shared_lib_2 = load_shared_lib (argv[2]);
  hip_module_up hip_module_1 = load_hip_module (argv[3]);
  hip_module_up hip_module_2 = load_hip_module (argv[4]);

  host_function_t shared_lib_1_sym
    = lookup_symbol (shared_lib_1, "func_host_shared_lib_1");
  host_function_t shared_lib_2_sym
    = lookup_symbol (shared_lib_2, "func_host_shared_lib_2");

  hipFunction_t code_object_1_kernel
    = lookup_symbol (hip_module_1, "func_device_code_object_1");
  hipFunction_t code_object_2_kernel
    = lookup_symbol (hip_module_2, "func_device_code_object_2");

  func_main (0);
  shared_lib_1_sym (0);
  shared_lib_2_sym (0);
  launch_kernel (code_object_1_kernel);
  launch_kernel (code_object_2_kernel);

  return 0;
}

#endif
