#include <stdint.h>

extern char foo[];

int
main ()
{
  return (uintptr_t) &foo == 0x12345678 ? 0 : 1;
}
