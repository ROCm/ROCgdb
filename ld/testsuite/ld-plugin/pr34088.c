/* A static LTO test to reference feclearexcept which is a compiler
   builtin function and isn't in the LTO symbol table when GCC is used.
   It triggers the run-time crash on glibc targets of a linker script
   libm.a without the fix for PR ld/34088 when GCC 13 or above is used:

   https://gcc.gnu.org/bugzilla/show_bug.cgi?id=124869

 */

#include <stdio.h>
#include <fenv.h>

int
main (void)
{
  feclearexcept (FE_ALL_EXCEPT);
  printf ("PASS\n");
  return 0;
}
