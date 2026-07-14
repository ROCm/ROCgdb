/* This testcase is part of GDB, the GNU debugger.

   Copyright 2026 Free Software Foundation, Inc.

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

/* C++ variant of the Fortran example from PR34034.  */

namespace mod_a {
  int xxx = 10;
}

namespace mod_b {
  using namespace mod_a;
  int yyy = 20;
}

static void foo () {}

int
main (void)
{
  foo ();	/* main-entry.  */

  { /* Variant 1: using block is stop block, using block is not function block.  */
    using namespace mod_b;
    void (xxx + yyy);
    foo ();	/* main-1.  */
  }

  { /* Variant 2: using block is super block of stop block, using block is not function block.  */
    using namespace mod_b;
    {
      void (xxx + yyy);
      foo ();	/* main-2.  */
    }
  }

  using namespace mod_b;

  { /* Variant 3: using block is super block of stop block, using block is function block.  */
    void (xxx + yyy);
    foo ();	/* main-3.  */
  }

  /* Variant 4: using block is stop block, using block is function block.  */
  void (xxx + yyy);
  foo ();	/* main-4.  */

  return 0;
}
