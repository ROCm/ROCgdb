int
f1 (void)
{
  return 42;
}
int
f2 (void)
{
  return 47;
}

void *
fx (void)
{
  return f1;
}

int f (void) __attribute__ ((ifunc ("fx")));

int
main ()
{
  int (*p) () = f;
  asm ("# prevent optimization" : "+r"(p));
  __builtin_printf ("%d\n", p ());
  return 0;
}
