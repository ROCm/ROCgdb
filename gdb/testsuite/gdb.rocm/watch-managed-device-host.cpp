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

#include <hip/hip_runtime.h>
#include <string.h>
#include <stdio.h>
#include <iostream>
#include "rocm-test-utils.h"

__global__ void
kern (int *ptr)
{
  *ptr = 1;
  WAIT_MEM;
}

int *managed_ptr = nullptr;

enum residency_mode
{
  /* Bias toward VRAM, CPU-accessible.  */
  DEVICE,
  /* Bias toward system memory, GPU-accessible.  */
  HOST,
};

/* Like CHECK, but don't error out if the arguments to the call are
   unsupported.  Used with the hipMemAdvise hints below.  */
#define CHECK_IGNORE_UNSUPPORTED(cmd)					\
  do									\
    {									\
      hipError_t err = cmd;						\
      if (err != hipSuccess && err != hipErrorInvalidValue)		\
	CHECK (err);							\
    }									\
  while (0)

/* The hipMemAdvise hints may not be supported on Windows.  */
#ifdef _WIN32
# define CHECK_MAYBE_IGNORE_UNSUPPORTED CHECK_IGNORE_UNSUPPORTED
#else
# define CHECK_MAYBE_IGNORE_UNSUPPORTED CHECK
#endif

/* Apply RESIDENCY to PTR.  */

static void
apply_residency (int *ptr, enum residency_mode residency)
{
  constexpr int device_id = 0;

  int resident_id;
  int other_id;
  if (residency == DEVICE)
    {
      resident_id = device_id;
      other_id = hipCpuDeviceId;
    }
  else
    {
      resident_id = hipCpuDeviceId;
      other_id = device_id;
    }

  CHECK_MAYBE_IGNORE_UNSUPPORTED
    (hipMemAdvise (ptr, sizeof (int), hipMemAdviseSetPreferredLocation,
		   resident_id));

  CHECK_MAYBE_IGNORE_UNSUPPORTED
    (hipMemAdvise (ptr, sizeof (int), hipMemAdviseSetAccessedBy,
		   other_id));

  /* Establish residency on RESIDENT_ID device.  */
  CHECK (hipMemPrefetchAsync (ptr, sizeof (int), resident_id, 0));
  CHECK (hipDeviceSynchronize ());
}

int
main (int argc, char **argv)
{
  if (argc != 2)
    {
      std::cerr
	<< "Usage: " << argv[0] << " device|host" << std::endl;
      return EXIT_FAILURE;
    }

  const char *residency_str = argv[1];
  residency_mode residency;

  if (strcmp (residency_str, "device") == 0)
    residency = DEVICE;
  else if (strcmp (residency_str, "host") == 0)
    residency = HOST;
  else
    {
      std::cerr << "Unsupported residency " << residency_str << std::endl;
      return EXIT_FAILURE;
    }

  CHECK (hipMallocManaged (&managed_ptr, sizeof (int)));

  apply_residency (managed_ptr, residency); /* set break 1 here */

  /* Warm up GPU and trigger initial watchpoint in kernel.  */
  kern<<<1, 1>>> (managed_ptr);
  CHECK (hipDeviceSynchronize ());

  /* Re-establish residency in case debugger or kernel access caused
     migration.  */
  apply_residency (managed_ptr, residency); /* set break 2 here */

  printf ("Pointer address: %p\n", (void *) managed_ptr);
  printf ("Attempting host write (watchpoint expected)...\n");

  /* Trigger watchpoint from the host.  */
  *managed_ptr = 2;

  printf ("Write successful.  Value: %d\n", *managed_ptr);

  CHECK (hipFree (managed_ptr));
  return 0;
}
