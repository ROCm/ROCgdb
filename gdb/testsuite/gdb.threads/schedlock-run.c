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
#include <unistd.h>

static void *
startup_thread_func (void *arg)
{
  return arg;
}

static void *
late_thread_func (void *arg)
{
  return arg;
}

int
main (void)
{
  pthread_t thread;

  pthread_create (&thread, NULL, startup_thread_func, NULL);
  pthread_join (thread, NULL);

  pthread_create (&thread, NULL, late_thread_func, NULL); /* break here */

  /* With scheduler-locking on, the late thread is held stopped, so it
     can't be joined.  Sleep long enough that, if it is scheduled, it
     hits the late_thread_func breakpoint before main returns.  */
  sleep (3);

  return 0; /* break after late thread */
}
