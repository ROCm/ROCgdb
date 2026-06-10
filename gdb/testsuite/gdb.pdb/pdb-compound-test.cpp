/* Copyright (C) 2026 Free Software Foundation, Inc.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

/* PDB Compound Type Test Program — exercises struct, class, union, enum
   type records and fieldlist sub-records including LF_INDEX continuation.

   Compile with clang-cl:
     clang-cl /Zi /Od /Fe:pdb-compound-test.exe pdb-compound-test.cpp

   This test program is used by:
     pdb-compound-symbols.exp   — info pdb-symbols checks
     pdb-compound-info.exp      — ptype / print field checks  */

#include <cstdio>

/* ================================================================
   Enum types (LF_ENUM + LF_ENUMERATE sub-records).
   ================================================================ */

/* Basic enum — small values.  */
enum Color { Red = 0, Green = 1, Blue = 2, Yellow = 3, Black = 255 };

/* Scoped enum — underlying type is int.  */
enum class Direction { North = 0, South = 1, East = 2, West = 3 };

/* Enum with large values — tests numeric leaf encoding.  */
enum BigValues {
  BV_ZERO    = 0,
  BV_SMALL   = 42,
  BV_MEDIUM  = 1000,
  BV_LARGE   = 70000,
  BV_HUGE    = 100000
};

/* ================================================================
   Plain struct (LF_STRUCTURE).
   ================================================================ */

struct SimpleStruct {
  int         x;
  int         y;
  double      value;
  char        tag;
  const char *name;
};

/* ================================================================
   Struct with bitfields (LF_BITFIELD).
   ================================================================ */

struct PackedFlags {
  unsigned int visible : 1;
  unsigned int enabled : 1;
  unsigned int mode    : 4;
  unsigned int level   : 8;
  unsigned int id      : 18;
};

/* ================================================================
   Struct with many members — ensures fieldlist is large enough
   to test LF_INDEX continuation (> 64 KB fieldlist data).
   Each field is 12+ bytes in the fieldlist (attr + type_ti + numeric
   offset + name + padding).  ~350 fields should exceed 64 KB.
   ================================================================ */

struct HugeStruct {
  /* Block 1 — fields f000..f099 (int).  */
  int f000; int f001; int f002; int f003; int f004;
  int f005; int f006; int f007; int f008; int f009;
  int f010; int f011; int f012; int f013; int f014;
  int f015; int f016; int f017; int f018; int f019;
  int f020; int f021; int f022; int f023; int f024;
  int f025; int f026; int f027; int f028; int f029;
  int f030; int f031; int f032; int f033; int f034;
  int f035; int f036; int f037; int f038; int f039;
  int f040; int f041; int f042; int f043; int f044;
  int f045; int f046; int f047; int f048; int f049;
  int f050; int f051; int f052; int f053; int f054;
  int f055; int f056; int f057; int f058; int f059;
  int f060; int f061; int f062; int f063; int f064;
  int f065; int f066; int f067; int f068; int f069;
  int f070; int f071; int f072; int f073; int f074;
  int f075; int f076; int f077; int f078; int f079;
  int f080; int f081; int f082; int f083; int f084;
  int f085; int f086; int f087; int f088; int f089;
  int f090; int f091; int f092; int f093; int f094;
  int f095; int f096; int f097; int f098; int f099;

  /* Block 2 — fields f100..f199 (double).  */
  double f100; double f101; double f102; double f103; double f104;
  double f105; double f106; double f107; double f108; double f109;
  double f110; double f111; double f112; double f113; double f114;
  double f115; double f116; double f117; double f118; double f119;
  double f120; double f121; double f122; double f123; double f124;
  double f125; double f126; double f127; double f128; double f129;
  double f130; double f131; double f132; double f133; double f134;
  double f135; double f136; double f137; double f138; double f139;
  double f140; double f141; double f142; double f143; double f144;
  double f145; double f146; double f147; double f148; double f149;
  double f150; double f151; double f152; double f153; double f154;
  double f155; double f156; double f157; double f158; double f159;
  double f160; double f161; double f162; double f163; double f164;
  double f165; double f166; double f167; double f168; double f169;
  double f170; double f171; double f172; double f173; double f174;
  double f175; double f176; double f177; double f178; double f179;
  double f180; double f181; double f182; double f183; double f184;
  double f185; double f186; double f187; double f188; double f189;
  double f190; double f191; double f192; double f193; double f194;
  double f195; double f196; double f197; double f198; double f199;

  /* Block 3 — fields f200..f299 (char).  */
  char f200; char f201; char f202; char f203; char f204;
  char f205; char f206; char f207; char f208; char f209;
  char f210; char f211; char f212; char f213; char f214;
  char f215; char f216; char f217; char f218; char f219;
  char f220; char f221; char f222; char f223; char f224;
  char f225; char f226; char f227; char f228; char f229;
  char f230; char f231; char f232; char f233; char f234;
  char f235; char f236; char f237; char f238; char f239;
  char f240; char f241; char f242; char f243; char f244;
  char f245; char f246; char f247; char f248; char f249;
  char f250; char f251; char f252; char f253; char f254;
  char f255; char f256; char f257; char f258; char f259;
  char f260; char f261; char f262; char f263; char f264;
  char f265; char f266; char f267; char f268; char f269;
  char f270; char f271; char f272; char f273; char f274;
  char f275; char f276; char f277; char f278; char f279;
  char f280; char f281; char f282; char f283; char f284;
  char f285; char f286; char f287; char f288; char f289;
  char f290; char f291; char f292; char f293; char f294;
  char f295; char f296; char f297; char f298; char f299;

  /* Block 4 — fields f300..f399 (long long).  */
  long long f300; long long f301; long long f302; long long f303;
  long long f304; long long f305; long long f306; long long f307;
  long long f308; long long f309; long long f310; long long f311;
  long long f312; long long f313; long long f314; long long f315;
  long long f316; long long f317; long long f318; long long f319;
  long long f320; long long f321; long long f322; long long f323;
  long long f324; long long f325; long long f326; long long f327;
  long long f328; long long f329; long long f330; long long f331;
  long long f332; long long f333; long long f334; long long f335;
  long long f336; long long f337; long long f338; long long f339;
  long long f340; long long f341; long long f342; long long f343;
  long long f344; long long f345; long long f346; long long f347;
  long long f348; long long f349; long long f350; long long f351;
  long long f352; long long f353; long long f354; long long f355;
  long long f356; long long f357; long long f358; long long f359;
  long long f360; long long f361; long long f362; long long f363;
  long long f364; long long f365; long long f366; long long f367;
  long long f368; long long f369; long long f370; long long f371;
  long long f372; long long f373; long long f374; long long f375;
  long long f376; long long f377; long long f378; long long f379;
  long long f380; long long f381; long long f382; long long f383;
  long long f384; long long f385; long long f386; long long f387;
  long long f388; long long f389; long long f390; long long f391;
  long long f392; long long f393; long long f394; long long f395;
  long long f396; long long f397; long long f398; long long f399;

  /* Block 5 — fields f400..f499 (float).  */
  float f400; float f401; float f402; float f403; float f404;
  float f405; float f406; float f407; float f408; float f409;
  float f410; float f411; float f412; float f413; float f414;
  float f415; float f416; float f417; float f418; float f419;
  float f420; float f421; float f422; float f423; float f424;
  float f425; float f426; float f427; float f428; float f429;
  float f430; float f431; float f432; float f433; float f434;
  float f435; float f436; float f437; float f438; float f439;
  float f440; float f441; float f442; float f443; float f444;
  float f445; float f446; float f447; float f448; float f449;
  float f450; float f451; float f452; float f453; float f454;
  float f455; float f456; float f457; float f458; float f459;
  float f460; float f461; float f462; float f463; float f464;
  float f465; float f466; float f467; float f468; float f469;
  float f470; float f471; float f472; float f473; float f474;
  float f475; float f476; float f477; float f478; float f479;
  float f480; float f481; float f482; float f483; float f484;
  float f485; float f486; float f487; float f488; float f489;
  float f490; float f491; float f492; float f493; float f494;
  float f495; float f496; float f497; float f498; float f499;
};

/* ================================================================
   Union (LF_UNION).
   ================================================================ */

union MultiValue {
  int         as_int;
  float       as_float;
  double      as_double;
  char        as_bytes[8];
  long long   as_longlong;
};

/* ================================================================
   Base class with virtual methods (LF_CLASS + LF_BCLASS +
   LF_ONEMETHOD / LF_METHOD + LF_VFUNCTAB).
   ================================================================ */

class Base {
public:
  int base_id;

  Base () : base_id (0) {}
  virtual ~Base () {}

  virtual int compute () const { return base_id; }
  virtual const char *name () const { return "Base"; }

  int get_base_id () const { return base_id; }
};

/* ================================================================
   Derived class — virtual overrides (tests LF_BCLASS in fieldlist,
   LF_ONEMETHOD with CV_MPROP_VIRTUAL / CV_MPROP_INTRO).
   ================================================================ */

class Derived : public Base {
public:
  double extra;
  static int instance_count;

  Derived () : extra (0.0) { instance_count++; }
  explicit Derived (double e) : extra (e) { instance_count++; }
  ~Derived () override { instance_count--; }

  int compute () const override { return base_id + (int)extra; }
  const char *name () const override { return "Derived"; }

  double get_extra () const { return extra; }

  /* Multiple overloads — forces LF_METHOD + LF_METHODLIST.  */
  void set_extra (double v) { extra = v; }
  void set_extra (int v) { extra = (double)v; }
  void set_extra (float v) { extra = (double)v; }
};

int Derived::instance_count = 0;

/* ================================================================
   Abstract class — CV_MPROP_PUREINTRO (= 0 introducing virtual).
   ================================================================ */

class AbstractShape {
public:
  virtual ~AbstractShape () {}

  /* Pure virtual introducing methods — CV_MPROP_PUREINTRO.  */
  virtual double area () const = 0;
  virtual const char *shape_name () const = 0;
};

/* ================================================================
   Concrete derived from abstract — CV_MPROP_PUREVIRT (pure virtual
   override, not the first declaration).
   ================================================================ */

class Circle : public AbstractShape {
public:
  double radius;

  Circle () : radius (0.0) {}
  explicit Circle (double r) : radius (r) {}

  double area () const override { return 3.14159 * radius * radius; }
  const char *shape_name () const override { return "Circle"; }
};

/* ================================================================
   Virtual base classes — LF_VBCLASS / LF_IVBCLASS.
   Diamond inheritance forces the compiler to emit both.
   ================================================================ */

class VBase {
public:
  int vb_data;
  VBase () : vb_data (0) {}
};

/* Virtual inheritance from VBase — LF_VBCLASS in Left's fieldlist.  */
class Left : virtual public VBase {
public:
  int left_val;
  Left () : left_val (0) {}
};

/* Virtual inheritance from VBase — LF_VBCLASS in Right's fieldlist.  */
class Right : virtual public VBase {
public:
  int right_val;
  Right () : right_val (0) {}
};

/* Diamond: inherits Left and Right, each virtually inheriting VBase.
   The compiler emits LF_IVBCLASS for the indirect virtual base VBase
   seen through Left and Right.  */
class Diamond : public Left, public Right {
public:
  int diamond_val;
  Diamond () : diamond_val (0) {}
};

/* ================================================================
   Class with nested type (LF_NESTTYPE), static member (LF_STMEMBER),
   private/protected members (accessibility testing).
   ================================================================ */

class Container {
public:
  /* Nested struct — LF_NESTTYPE in Container's fieldlist.  */
  struct Entry {
    int key;
    double val;
  };

  /* Nested enum — another LF_NESTTYPE.  */
  enum Status { Empty = 0, Partial = 1, Full = 2 };

  Container () : size_ (0), status_ (Empty) {}

  void add (const Entry &e) {
    if (size_ < 10) {
      entries_[size_++] = e;
      status_ = (size_ == 10) ? Full : Partial;
    }
  }

  int size () const { return size_; }
  Status status () const { return status_; }

  static int max_size () { return 10; }

private:
  Entry entries_[10];
  int size_;

protected:
  Status status_;
};

/* ================================================================
   Global variables of each compound type — these produce S_GDATA32
   / S_LDATA32 records whose type indices point to the compound
   type records we want to exercise.
   ================================================================ */

Color              g_color = Green;
Direction          g_dir = Direction::East;
BigValues          g_bv = BV_LARGE;

SimpleStruct       g_simple = { 10, 20, 3.14, 'A', "hello" };
MultiValue         g_union;
PackedFlags        g_flags = { 1, 1, 5, 200, 12345 };

Base               g_base;
Derived            g_derived (42.5);
Container          g_container;
Circle             g_circle (5.0);
Diamond            g_diamond;

/* Static: s_struct uses a struct type.  */
static SimpleStruct s_struct = { 100, 200, 2.71, 'Z', "static" };

/* A HugeStruct global — forces the large fieldlist (LF_INDEX).  */
HugeStruct         g_huge;

/* ================================================================
   Functions that use the types — so the compiler doesn't optimize
   them away and they appear in symbol records.
   ================================================================ */

int
use_enum (Color c, Direction d)
{
  return (int)c + (int)d;
}

double
use_struct (const SimpleStruct &s)
{
  return s.x + s.y + s.value;
}

unsigned int
use_flags (const PackedFlags &f)
{
  return f.visible + f.enabled + f.mode + f.level + f.id;
}

long long
use_union (const MultiValue &u)
{
  return u.as_longlong;
}

int
use_class (const Base &b)
{
  return b.compute ();
}

int
use_derived (const Derived &d)
{
  return d.compute () + (int)d.extra;
}

void
use_container (Container &c)
{
  Container::Entry e = { 1, 2.0 };
  c.add (e);
}

int
use_huge (const HugeStruct &h)
{
  return h.f000 + h.f099 + h.f199 + (int)h.f299
       + (int)h.f399 + (int)h.f499;
}

double
use_abstract (const AbstractShape &s)
{
  return s.area ();
}

int
use_diamond (const Diamond &d)
{
  return d.vb_data + d.left_val + d.right_val + d.diamond_val;
}

/* Marker for compound-info test to break at.  */
void
compound_marker ()
{
}

/* ================================================================
   main — instantiate everything so all types appear in the PDB.
   ================================================================ */

int
main ()
{
  /* Local variables of each compound type.  */
  Color local_color = Blue;
  Direction local_dir = Direction::West;
  SimpleStruct local_struct = { 1, 2, 0.5, 'L', "local" };
  MultiValue local_union;
  local_union.as_double = 99.9;

  Derived local_derived (7.7);
  Base *base_ptr = &local_derived;

  Container local_container;
  Container::Entry ent = { 42, 3.14 };
  local_container.add (ent);

  HugeStruct local_huge;
  local_huge.f000 = 1;
  local_huge.f099 = 2;
  local_huge.f199 = 3;
  local_huge.f299 = 4;
  local_huge.f399 = 5;
  local_huge.f499 = 6;

  Circle local_circle (3.0);
  Diamond local_diamond;
  local_diamond.vb_data = 42;
  local_diamond.left_val = 10;
  local_diamond.right_val = 20;
  local_diamond.diamond_val = 30;

  /* Use globals so they aren't optimized away.  */
  g_union.as_int = 0x12345678;
  g_base.base_id = 10;
  g_huge.f000 = 100;
  g_huge.f499 = 200;

  /* Call all functions to generate symbol records.  */
  int r1 = use_enum (local_color, local_dir);
  double r2 = use_struct (local_struct);
  long long r3 = use_union (local_union);
  int r4 = use_class (*base_ptr);
  int r5 = use_derived (local_derived);
  use_container (local_container);
  int r6 = use_huge (local_huge);
  unsigned int r7 = use_flags (g_flags);
  double r8 = use_abstract (local_circle);
  int r9 = use_diamond (local_diamond);

  /* Break here for compound-info field inspection.  */
  compound_marker ();

  printf ("r1=%d r2=%.2f r3=%lld r4=%d r5=%d r6=%d r7=%u "
	  "r8=%.2f r9=%d inst=%d s_x=%d g_bv=%d\n",
	  r1, r2, r3, r4, r5, r6, r7, r8, r9,
	  Derived::instance_count, s_struct.x, (int)g_bv);

  return 0;
}
