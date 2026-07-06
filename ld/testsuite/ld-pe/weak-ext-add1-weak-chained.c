/* Weak definition of func (adds 1), plus dummy that calls func.  */
extern int value;

__attribute__((weak)) void func (void)
{
  value += 1;
}

void
dummy (void)
{
  func ();
}
