void foo(void) asm("foo");
void foo2(void) asm("foo2");
int main(void) asm("main");

int main(void)
{
   foo ();

   for (;;)
	;
}

int a;

void foo1(void)
{
	a = 1;
}

void foo2(void)
{
	a = 2;
}

void foo3(void)
{
	a = 3;
}
