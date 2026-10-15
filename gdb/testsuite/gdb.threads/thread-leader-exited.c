/* This testcase is part of GDB, the GNU debugger.

   Copyright 2014-2026 Free Software Foundation, Inc.

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

static pthread_t main_thread;
static pthread_barrier_t barrier;

/* First worker.  */

static void *
start (void *arg)
{
  int i = pthread_join (main_thread, NULL);
  assert (i == 0);

  i = pthread_barrier_wait (&barrier);
  assert (i == 0 || i == PTHREAD_BARRIER_SERIAL_THREAD);

  return arg; /* break-here */
}

/* Second worker.  */

static void *
start_second (void *arg)
{
  int i = pthread_barrier_wait (&barrier);
  assert (i == 0 || i == PTHREAD_BARRIER_SERIAL_THREAD);

  return arg; /* break-here-second */
}

int
main (void)
{
  pthread_t thread, thread_second;

  main_thread = pthread_self ();

  int i = pthread_barrier_init (&barrier, NULL, 2);
  assert (i == 0);

  i = pthread_create (&thread, NULL, start, NULL);
  assert (i == 0);

  i = pthread_create (&thread_second, NULL, start_second, NULL);
  assert (i == 0);

  pthread_exit (NULL);
  assert (0);
  return 0;
}
