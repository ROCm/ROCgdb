#include <stdio.h>

extern long *get_ld_pcaddi(void);
extern long *get_ld(void);
extern long *get_gd_pcaddi(void);
extern long *get_gd(void);
extern long *get_desc_pcaddi(void);
extern long *get_desc(void);

int main(void)
{
  long *p = get_ld_pcaddi();
  if (*p != 100)
    return 1;

  p = get_ld();
  if (*p != 100)
    return 1;

  p = get_gd_pcaddi();
  if (*p != 100)
    return 1;

  p = get_gd();
  if (*p != 100)
    return 1;

  p = get_desc_pcaddi();
  if (*p != 100)
    return 1;

  p = get_desc();
  if (*p != 100)
    return 1;

  puts("PASS");
  return 0;
}
