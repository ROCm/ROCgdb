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
#include <time.h>

#define NCALLS 10
#define NSPIN 2

volatile int watched_global;
volatile int stop_spin;
volatile unsigned long spins[NSPIN];

/* Always runnable, so that GDB single-steps these threads alongside
   main while the software watchpoint exists.  This makes each pass
   through the vdso time read loop in main span several timer ticks.  */

static void *
spinner (void *arg)
{
  long i = (long) arg;

  while (!stop_spin)
    spins[i]++;
  return NULL;
}

void __attribute__ ((noinline))
marker (void)
{
  __asm__ volatile ("" ::: "memory");
}

int
main (void)
{
  pthread_t th[NSPIN];
  struct timespec ts;
  long i;

  for (i = 0; i < NSPIN; i++)
    pthread_create (&th[i], NULL, spinner, (void *) i);

  marker ();

  /* clock_gettime runs in the vdso, which retries its read when the
     kernel updates the time data meanwhile.  */
  for (i = 0; i < NCALLS; i++)
    clock_gettime (CLOCK_MONOTONIC, &ts);

  watched_global = 1;

  stop_spin = 1;
  for (i = 0; i < NSPIN; i++)
    pthread_join (th[i], NULL);

  return 0;
}
