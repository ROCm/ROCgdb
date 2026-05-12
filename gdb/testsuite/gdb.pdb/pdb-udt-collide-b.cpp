/* Fixture (TU 2 of 2) for pdb-udt-collide.exp.  Binds `StructAlias' to
   BStruct; see pdb-udt-collide-a.cpp.  */

struct BStruct
{
  long long b;
  double c;
};

using StructAlias = BStruct;

/* Referenced so the type is emitted.  */
StructAlias g_b;

int
use_b ()
{
  g_b.b = 2;
  return (int) g_b.b; /* BREAK HERE (b)  */
}
