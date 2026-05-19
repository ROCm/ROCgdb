/* This testcase is part of GDB, the GNU debugger.

   Copyright (C) 2026 Free Software Foundation, Inc.
   Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

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

/* The parent process forks N children (N comes from argv[1], default
   16).  Each child re-execs itself so that its GPU runtime is
   initialized in a clean address space, then launches its own GPU
   kernel.  Re-exec'ing makes each child a separate process from the
   kernel-driver / debug API point of view, which is what exercises
   the multi-process GPU debug path under load.  */

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <hip/hip_runtime.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocm-test-utils.h"

/* Default fan-out when no count is given on the command line.  Picked
   so it crosses the typical 8-process threshold for simultaneous
   debug-attached GPU processes.  */
#define DEFAULT_NUM_CHILDREN 16

__global__ void
kern ()
{
  asm ("s_sleep 1");
}

static int
child (int argc, char **argv)
{
  if (argc < 4)
    {
      fprintf (stderr, "child: expected: %s child <idx> <num_devices>\n",
	       argv[0]);
      return -1;
    }

  int idx = atoi (argv[2]);
  int num_devices = atoi (argv[3]);
  if (num_devices <= 0)
    {
      fprintf (stderr, "child %d: invalid num_devices %d\n", idx, num_devices);
      return -1;
    }

  CHECK (hipSetDevice (idx % num_devices));
  kern<<<1, 1>>> ();
  CHECK (hipDeviceSynchronize ());
  return 0;
}

static int
parent (char **argv, int num_children)
{
  int num_devices;
  CHECK (hipGetDeviceCount (&num_devices));
  if (num_devices <= 0)
    {
      fprintf (stderr, "no GPU devices available\n");
      return -1;
    }

  /* Break here.  */

  for (int i = 0; i < num_children; i++)
    {
      char idx_buf[32] = {};
      char ndev_buf[32] = {};
      snprintf (idx_buf, sizeof (idx_buf), "%d", i);
      snprintf (ndev_buf, sizeof (ndev_buf), "%d", num_devices);

      pid_t pid = fork ();
      if (pid == -1)
	{
	  perror ("fork");
	  return -1;
	}

      if (pid == 0)
	{
	  if (execl (argv[0], argv[0], "child", idx_buf, ndev_buf,
		     (char *) nullptr) == -1)
	    {
	      perror ("execl");
	      _exit (127);
	    }
	}
    }

  /* Reap every child.  Any non-zero exit from a child is a stress-test
     failure (e.g. a runtime initialization failure under contention).  */
  int failed = 0;
  while (true)
    {
      int ws;
      pid_t ret = waitpid (-1, &ws, 0);
      if (ret == -1 && errno == ECHILD)
	break;
      if (ret > 0 && (!WIFEXITED (ws) || WEXITSTATUS (ws) != 0))
	failed++;
    }

  /* Last break here.  */
  return failed == 0 ? 0 : 1;
}

int
main (int argc, char **argv)
{
  if (argc >= 2 && strcmp (argv[1], "child") == 0)
    return child (argc, argv);

  int num_children = DEFAULT_NUM_CHILDREN;
  if (argc >= 2)
    {
      num_children = atoi (argv[1]);
      if (num_children <= 0)
	num_children = DEFAULT_NUM_CHILDREN;
    }

  return parent (argv, num_children);
}
