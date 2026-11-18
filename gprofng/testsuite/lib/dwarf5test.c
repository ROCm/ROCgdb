/* Test program for dwarf5.exp: the hot function is declared in this file and
   inlines code from dwarf5decl.h.  Clang describes such a function with
   DW_AT_decl_file 0, the primary source file of the DWARF 5 file table.  */
#include <stdio.h>
#include <time.h>
#include "dwarf5decl.h"

typedef long long hrtime_t;

static hrtime_t
gethrtime (void)
{
  struct timespec tp;
  hrtime_t rc = 0;
#ifdef CLOCK_MONOTONIC_RAW
  int r = clock_gettime (CLOCK_MONOTONIC_RAW, &tp);
#else
  int r = clock_gettime (CLOCK_MONOTONIC, &tp);
#endif

  if (r == 0)
    rc = ((hrtime_t) tp.tv_sec) * 1e9 + (hrtime_t) tp.tv_nsec;
  return rc;
}

volatile long x;

__attribute__ ((noinline)) void
hot_function (void)
{
  for (int j = 0; j < 1000000; j++)
    x = step (x);
}

int
main (void)
{
  hrtime_t start = gethrtime ();

  do
    hot_function ();
  while (start + 1e9 > gethrtime ());
  printf ("x=%ld\n", x);
  return 0;
}
