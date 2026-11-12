#include <stdio.h>

extern long *get_ld_pcaddi(void);
extern long *get_gd_pcaddi(void);
extern long *get_desc_pcaddi(void);

int main(void)
{
  long *p = get_ld_pcaddi();
  if (*p != 100)
    return 1;

  p = get_gd_pcaddi();
  if (*p != 100)
    return 1;

  p = get_desc_pcaddi();
  if (*p != 100)
    return 1;

  puts("PASS");
  return 0;
}
