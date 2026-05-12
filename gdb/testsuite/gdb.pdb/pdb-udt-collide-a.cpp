/* Fixture (TU 1 of 2) for pdb-udt-collide.exp.  Binds `StructAlias' to
   AStruct; pdb-udt-collide-b.cpp binds the same name to a different type.
   Only one of the two survives in the global symbol stream.  */

struct AStruct
{
  int a;
};

using StructAlias = AStruct;

/* Referenced so the type is emitted.  */
StructAlias g_a;

int
use_a ()
{
  g_a.a = 1;
  return g_a.a; /* BREAK HERE (a)  */
}
