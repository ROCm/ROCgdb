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

#include <unistd.h>
#include <pthread.h>

static volatile int spin = 1;

static void *
thread_func (void *arg)
{
  /* Unleash the main thread.  */
  spin = 0;

  for (;;)
    sleep (1);

  return NULL;
}

static void
breakpoint_1 (void)
{
  /* Do nothing.  */
}

static void
breakpoint_2 (void)
{
  /* Do nothing.  */
}

int
main ()
{
  pthread_t thread;
  pthread_create (&thread, NULL, thread_func, NULL);

  alarm (30);

  /* Make sure the thread is up an running.  */
  while (spin)
    sleep (1);

  breakpoint_1 ();
  int a = 42;
  breakpoint_2 ();

  pthread_join (thread, NULL);

  return 0;
}
