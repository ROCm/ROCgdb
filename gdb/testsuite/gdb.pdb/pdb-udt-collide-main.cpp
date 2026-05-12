/* Fixture (main TU) for pdb-udt-collide.exp.  Links the two sibling TUs
   that each bind `StructAlias' to a different type.  */

int use_a ();
int use_b ();

int
main ()
{
  return use_a () + use_b ();
}
