#include <stdio.h>

typedef void Fn();

void __attribute__((visibility("hidden")))
fun (void)
{}

extern void fun_public() __attribute__((alias("fun")));

void
call_callback (Fn *callback)
{
  if (callback == fun)
    printf("PASS\n");
  else
    printf("FAIL\n");

  callback ();
}
