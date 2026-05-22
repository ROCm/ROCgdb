extern __thread int x
__attribute__((visibility("hidden")))
__attribute__((weak));
extern __thread int x_used
__attribute__((visibility("hidden")))
__attribute__((weak));

int
main (void)
{
  if (!x_used)
    x++;
  return 0;
}
