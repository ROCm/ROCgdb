/* Second weak definition of func (adds 1) plus expected value.  */
extern int value;

__attribute__((weak)) void func (void)
{
  value += 1;
}

int expected = 3;
