/* Copyright (C) 2022 Advanced Micro Devices, Inc. All rights reserved.

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
#include <unistd.h>
#include <stdlib.h>

__device__ static void
add_one (int *out)
{
  ++*out;
}

__global__ void
the_kernel (int *out)
{
  *out = 0;

  for (int i = 0; i < 100; ++i)
    add_one (out);
}

int
main ()
{
  /* Make sure we don't run forever.  */
  alarm (30);

  int *result_ptr, result;
  hipError_t error = hipMalloc (&result_ptr, sizeof (int));
  if (error != hipSuccess)
    abort ();

  the_kernel<<<dim3(1), dim3(1), 0, 0>>> (result_ptr);

  error = hipMemcpyDtoH (&result, result_ptr, sizeof (int));
  if (error != hipSuccess)
    abort ();

  if (result != 100)
    abort ();

  /* Write that file so the test knows this program has successfully
     ran to completion after the detach.  */
  FILE *f = fopen (TOUCH_FILE_PATH, "w");
  if (f == nullptr)
    abort ();

  fclose (f);

  return 0;
}
