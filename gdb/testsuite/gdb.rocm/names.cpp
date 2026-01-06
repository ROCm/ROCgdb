/* This testcase is part of GDB, the GNU debugger.

   Copyright 2025-2026 Free Software Foundation, Inc.

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
#include <assert.h>
#include "gdb_watchdog.h"
#include <iostream>
#include <vector>
#include <unordered_map>
#include <string.h>

#define CHECK(cmd)                                                           \
  do                                                                         \
    {                                                                        \
      hipError_t error = cmd;                                                \
      if (error != hipSuccess)                                               \
	{                                                                    \
	  fprintf (stderr, "error: '%s'(%d) at %s:%d\n",                     \
		   hipGetErrorString (error), error, __FILE__, __LINE__);    \
	  exit (EXIT_FAILURE);                                               \
	}                                                                    \
    } while (0)

/* All kernels call this.  */

__device__ static void
wait_forever ()
{
  while (1)
    __builtin_amdgcn_s_sleep (1);
}

/* A kernel with a name longer than GDB's longest possible name.  */

__global__ void
long_kernel_name_abcdefghijklmnopqrstuvwxyz ()
{
  wait_forever ();
}

/* A kernel with a name longer than GDB's longest possible name.  */

__global__ void
long_kernel_name_0123456789 ()
{
  wait_forever ();
}

/* A kernel with a normal, short name.  */

__global__ void
kern ()
{
  wait_forever ();
}

/* Same, but with some parameters.  */

__global__ void
kern (int)
{
  wait_forever ();
}

/* A kernel with extern "C" linkage, which doesn't include an argument
   list in the demangled name.  */

extern "C" __global__ void
c_kern ()
{
  wait_forever ();
}

/* A kernel in a namespace.  */

namespace NS {

__global__ void
kern_ns ()
{
  wait_forever ();
}

}

/* A kernel in an anonymous namespace.  */

namespace {

__global__ void
anonymous ()
{
  wait_forever ();
}

}

/* An enum with no enumerators to make the kernel symbol name have an
   open parenthesis in the template argument list, like 'void
   templ_non_type<(E)0>()'.  This ensures GDB does not confuse the
   template argument's opening parenthesis with the start of the
   function's parameter list.  */
enum E {};

template<E val>
__global__ void
templ_non_type ()
{
  wait_forever ();
}

template<typename... Args>
__global__ void
templ_type (Args... args)
{
  wait_forever ();
}

/* List of kernel names and a callback that launches the named
   kernel.  */
static struct
{
  /* The kernel name.  */
  const char *kernel;

  /* Launch kernel on stream.  */
  void (*launch) (hipStream_t);
} kernels[] = {
  {
    "long_kernel_name_abcdefghijklmnopqrstuvwxyz", [] (hipStream_t st)
    {
      long_kernel_name_abcdefghijklmnopqrstuvwxyz
	<<<dim3 (1), dim3 (1), 0, st>>> ();
    }
  },
  {
    "long_kernel_name_0123456789", [] (hipStream_t st)
    {
      long_kernel_name_0123456789
	<<<dim3 (1), dim3 (1), 0, st>>> ();
    }
  },
  {
    "kern", [] (hipStream_t st)
    {
      kern
	<<<dim3 (1), dim3 (1), 0, st>>> ();
    }
  },
  {
    "kern-1", [] (hipStream_t st)
    {
      kern
	<<<dim3 (1), dim3 (1), 0, st>>> (1);
    }
  },
  {
    "c_kern", [] (hipStream_t st)
    {
      c_kern
	<<<dim3 (1), dim3 (1), 0, st>>> ();
    }
  },
  {
    "kern_ns", [] (hipStream_t st)
    {
      NS::kern_ns
	<<<dim3 (1), dim3 (1), 0, st>>> ();
    }
  },
  {
    "anonymous", [] (hipStream_t st)
    {
      anonymous
	<<<dim3 (1), dim3 (1), 0, st>>> ();
    }
  },
  {
    "templ_non_type", [] (hipStream_t st)
    {
      templ_non_type<(E)0>
	<<<dim3 (1), dim3 (1), 0, st>>> ();
    }
  },
  {
    "templ_type", [] (hipStream_t st)
    {
      templ_type<int, long>
	<<<dim3 (1), dim3 (1), 0, st>>> (1, 2);
    }
  },
};

int
main (int argc, char **argv)
{
  /* So that the GPU threads don't spin forever.  */
  gdb_watchdog (30);

  if (argc != 2)
    {
      std::cerr
	<< "Usage: " << argv[0] << " KERNEL|all" << std::endl;
      return EXIT_FAILURE;
    }

  /* Launch each kernel on its own stream, so they can all run
     concurrently.  */
  std::vector<hipStream_t> streams;

  for (auto &k : kernels)
    {
      if (strcmp (argv[1], "all") == 0 || strcmp (argv[1], k.kernel) == 0)
	{
	  hipStream_t st;
	  CHECK (hipStreamCreate (&st));
	  streams.push_back (st);
	  k.launch (st);
	  CHECK (hipGetLastError ());
	}
    }

  CHECK (hipDeviceSynchronize ());

  for (hipStream_t &st : streams)
    CHECK (hipStreamDestroy (st));

  return 0;
}
