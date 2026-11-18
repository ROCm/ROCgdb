/* Test program for dwarf5.exp: one hot function that runs for about a second,
   so that the profile has samples attributed to a known source line.  */
#include <stdio.h>
#include <time.h>

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
    x = x + 1;
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
