#ifndef __PIC__
#error "this file must be compiled with -fPIC"
#endif

typedef void Fn();
void fun_public(void);
void call_callback(Fn *callback);

int
main ()
{
  fun_public ();
  call_callback (fun_public);
  return 0;
}
