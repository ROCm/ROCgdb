/* Strong undefined reference to func.  */
#include <stdio.h>

void func (void);
void dummy (void);

int value = 1;
extern int expected;

int
main (void)
{
  func ();
  dummy ();
  if (value != expected)
    {
      printf ("expected %d, got %d\n", expected, value);
      return 1;
    }
  return 0;
}
