int (*p)(void);
int main(void) asm("main");

int
main ()
{
  return p () != 0x1234;
}
