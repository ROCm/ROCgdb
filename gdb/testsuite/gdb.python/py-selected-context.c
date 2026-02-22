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
   along with this program.  If not, see  <http://www.gnu.org/licenses/>.  */

#include <pthread.h>

volatile int global_var = 0;

/* Thread inner function.  */

void
thread_breakpt (void)
{
  global_var = global_var + 1;	/* First breakpoint.  */
}

/* The thread entry point.  */

void *
worker_thread (void *unused)
{
  thread_breakpt ();
  return NULL;
}

/* Create a thread, and wait for it to complete.  */

void
run_thread (void)
{
  pthread_t thr;

  pthread_create (&thr, NULL, worker_thread, NULL);

  pthread_join (thr, NULL);
}

int
main (void)
{
  run_thread ();
  return 0;		/* Second breakpoint.  */
}
