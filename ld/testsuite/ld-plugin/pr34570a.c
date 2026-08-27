#include <stdint.h>

char foo[8];

int
main ()
{
  return (uintptr_t) &foo == 0x12345678 ? 0 : 1;
}
