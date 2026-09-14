#include <stdio.h>
#include <stdlib.h>

void
ProfilerStart (const char *filename)
{
  printf ("Starting with arg: %s!\n", filename);
}

static __attribute__ ((constructor)) void
startup (void)
{
  const char *v = getenv ("CPUPROFILE");
  if (!v)
    return;
  ProfilerStart (v);
}
