/* This testcase is part of GDB, the GNU debugger.
   Copyright 2024-2026 Free Software Foundation, Inc.
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

static void *
task (void *arg)
{
  volatile unsigned int duration = 0;

  int a = 1; /* worker thread break 1.  */

  sleep (duration);
  int b = 2; /* worker thread break 2.  */

  sleep (duration);
  return NULL;
}

int
main (void)
{
  pthread_t th;

  alarm (30);

  pthread_create (&th, NULL, task, NULL);
  pthread_join (th, NULL);

  return 0; /* main thread break.  */
}
