/* Weak definition of func in main (adds 4).  */
#include <stdio.h>

int value = 1;

__attribute__((weak)) void func (void)
{
  value += 4;
}

void dummy (void);
extern int expected;

int
main (void)
{
  if (func)
    func ();
  dummy ();
  if (value != expected)
    {
      printf ("expected %d, got %d\n", expected, value);
      return 1;
    }
  return 0;
}
