/* Copyright (C) 2020 Free Software Foundation, Inc.
   Copyright (C) 2020 Advanced Micro Devices, Inc. All rights reserved.

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
   along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <stdio.h>
#include <iostream>
#include <signal.h>
#include <unistd.h>
#include <hip/hip_runtime.h>

/* Kernel entry point.  Loop forever, while avoiding to peg the
   GPU.  */

__global__ void
kernel ()
{
  while (1)
    __builtin_amdgcn_s_sleep (1);
}

static void
sigalrm_handler (int signo)
{
  fprintf (stderr, "whoops, timeout\n");
  exit (1);
}

int
main (int argc, char **argv)
{
  hipLaunchKernelGGL (kernel, dim3 (1), dim3 (1),
		      0 /*dynamicShared*/, 0 /*stream*/);

  /* Don't run forever.  */
  signal (SIGALRM, sigalrm_handler);
  alarm (10);

  /* Wait until kernel finishes.  In this case, this blocks
     "forever".  */
  hipDeviceSynchronize ();

  return 0;
}
