/* This testcase is part of GDB, the GNU debugger.

   Copyright 2022-2024 Free Software Foundation, Inc.

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

static pthread_barrier_t barrier;

static void *
worker_func (void *ignored)
{
  pthread_barrier_wait (&barrier);
  return NULL;
}

void
started (void)
{
}

int
main (void)
{
  pthread_t worker_thread;
  pthread_barrier_init (&barrier, NULL, 2);

  pthread_create (&worker_thread, NULL, worker_func, NULL);

  started ();

  pthread_barrier_wait (&barrier);
  pthread_join (worker_thread, NULL);
  pthread_barrier_destroy (&barrier);

  return 0;
}
