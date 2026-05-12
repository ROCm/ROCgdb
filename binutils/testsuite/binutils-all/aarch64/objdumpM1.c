int datum = 22;

extern int func1 (int);
extern int func2 (int);
extern int func3 (int);
extern int func4 (int);
extern int func5 (int);

/* Create an array of function pointers in the .data section.  */
struct ptrs
{
  int (* fptr)(int);
  int field;
}
fred [5] =
{
  { func1, 1 },
  { func2, 2 },
  { func3, 3 },
  { func4, 4 },
  { func5, 5 }
};

int main (int arg)
{
  /* FIXME: We want a way to make sure that undefined instructions that
     are inserted via the .inst directive are not annotated.  (Since the .inst
     directive is used explicitly for instructions and the user does not care
     if they happen to match a symbolic value).
     
     Unfortunately the .inst directive only supports constant expressions so
     we cannot use an unresolved symbolic value.  Inserting a symbolic value
     via one of the data directives does not work since they are labeled as
     data (via the MAP_DATA mapping state) and hence they become eligible
     for annotation.

     So for now we are stuck.  */

  /* Stop the compiler from optimizing away the function pointer array
     by using it in a non-predictable manner.  */
  return fred[arg].fptr (fred[arg].field * datum);
}
