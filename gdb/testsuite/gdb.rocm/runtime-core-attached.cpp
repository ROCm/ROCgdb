/* Copyright (C) 2023-2026 Free Software Foundation, Inc.
   Copyright (C) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.

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
#include <thread>
#include <chrono>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "gdb_watchdog.h"

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

/* Pagefault kernel.  In this testcase, OUT contains an address not reachable
   by the GPU, triggering a page fault.  */

__global__ void
pagefault_kernel (int *out)
{
  int local = 42;
  *out = 8;
}

/* This kernel will call abort (s_trap 2), which should cause the runtime to
   generate a core dump.  */

__global__ void
abort_kernel ()
{
  int local = 42;
  abort ();
}

/* Secondary kernel, meant to run concurrently on a separate stream.  This
   kernel is meant to be running when the "main" kernel will generate an
   exception.  This is to ensure that GDB can load kernels which have raised an
   exception (and entered the trap handler) and kernels which have not.  */

__global__ void
aux_kernel ()
{
  int local = 72;

  while (true)
    __builtin_amdgcn_s_sleep (1);
}

enum testcase_t
{
  memfault,
  abort
};

int
main (int argc, char **argv)
{
  /* Make sure that the process terminates if the exception is not caught by
     the ROCr runtime.  */
  gdb_watchdog (30);

  if (argc != 2)
    {
      std::cerr
	<< "Usage: " << argv[0] << " pagefault|abort" << std::endl;
      return EXIT_FAILURE;
    }

  std::string teststr = argv[1];
  testcase_t test;
  if (teststr == "pagefault")
    test = testcase_t::memfault;
  else if (teststr == "abort")
    test = testcase_t::abort;
  else
    {
      std::cerr << "Invalid test name \"" << teststr << "\"" << std::endl;
      return EXIT_FAILURE;
    }

  hipStream_t st1;
  hipStream_t st2;

  CHECK (hipStreamCreate (&st1));
  CHECK (hipStreamCreate (&st2));

  aux_kernel<<<1, 1, 0, st1>>> ();

  /* Make sure that the aux kernel gets time to start.  */
  std::this_thread::sleep_for (std::chrono::seconds { 2 });

  switch (test)
    {
    case testcase_t::memfault:
      {
	int *out = nullptr;
	pagefault_kernel<<<1, 1, 0, st2>>> (out);
	break;
      }
    case testcase_t::abort:
      abort_kernel<<<1, 1, 0, st2>>> ();
      break;
    };

  CHECK (hipDeviceSynchronize ());
}
