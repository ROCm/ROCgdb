/* This testcase is part of GDB, the GNU debugger.

   Copyright 2025 Free Software Foundation, Inc.

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

#include "dwz-symtabs-foo.h"
#include "dwz-symtabs-common.h"

struct foo_data_type
{
  int positive_value;
  int negative_value;
};

static struct foo_data_type foo_data;

void
do_callback (foo_callback_type cb)
{
  cb ();	/* Call line.  */
}

void
process_foo_data (int input)
{
#ifdef FOO2
  int other_value = 4;
#else
  int other_value = 6;
#endif

  if (input > 0)
    foo_data.positive_value += add_some_int (input, other_value);
  else
    foo_data.negative_value += add_some_int (other_value, input);
}

/* This comment must appear at the end of the source file with no compiled
   code after it.  When looking for a line number LINENO, GDB looks for a
   symtab with a line table entry for LINENO or a later line.  It is
   important for this test that there be no suitable line table entry.  The
   numbers have no real meaning, I just want to ensure that the 'XXX' line,
   which is what the test lists is far from any previous source listing.

   0
   1
   2
   3
   4
   5
   6
   7
   8
   9
   10
   XXX: Second line to list.
   10
   9
   8
   7
   6
   5
   4
   3
   2
   1
   0 */
