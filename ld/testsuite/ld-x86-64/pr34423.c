#include <stdio.h>

int
bar0 (void)
{
  return 0;
}

int
bar1 (void)
{
  return 1;
}

int
bar2 (void)
{
  return 2;
}

int
bar3 (void)
{
  return 3;
}

int
bar4 (void)
{
  return 4;
}

extern int foo (int);

int
main ()
{
  if (foo (1) == 1
      && foo (3) == 3
      && foo (4) == 4
      && foo (2) == 2
      && foo (0) == 0)
    printf ("PASS\n");

  return 0;
}
