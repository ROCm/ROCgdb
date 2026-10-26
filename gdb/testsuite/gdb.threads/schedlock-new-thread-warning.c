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

#include <pthread.h>
#include <assert.h>
#include <unistd.h>

#define NUM_THREADS 2

static void *
thread_func (void *arg)
{
  /* Don't exit, so that GDB doesn't report thread exits after the main
     thread stops at the breakpoint, when scheduler-locking is off.  */
  while (1)
    sleep (1);

  return NULL;
}

int
main (void)
{
  pthread_t threads[NUM_THREADS];
  int i;

  for (i = 0; i < NUM_THREADS; i++)
    {
      int ret = pthread_create (&threads[i], NULL, thread_func, NULL);
      assert (ret == 0);
    }

  return 0; /* break here */
}
