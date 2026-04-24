/* Self tests for the iterator_range class.

   Copyright (C) 2026 Free Software Foundation, Inc.

   This file is part of GDB.

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

#include "gdbsupport/selftest.h"
#include "gdbsupport/iterator-range.h"
#include "int-array-iterator.h"

namespace selftests {

using int_array_iterator_range = iterator_range<int_array_iterator>;

static void
test_iterator_range_1 (int_array_iterator_range &r, int array[], int size,
		       int_array_iterator &begin, int_array_iterator &end)
{
  SELF_CHECK (r.begin () == begin);
  SELF_CHECK (r.end () == end);
  SELF_CHECK (r.size () == size);
  SELF_CHECK (r.empty () == (size == 0));

  int j = 0;
  for (auto i : r)
    {
      SELF_CHECK (j < size);
      SELF_CHECK (i == array[j]);
      j++;
    }
  SELF_CHECK (j == size);
}

static void
test_iterator_range ()
{
  int array[] = { 4, 4, 5, 6, 7, 8, 9 };
  int array_size = ARRAY_SIZE (array);

  int_array_iterator begin (array, array_size);
  int_array_iterator end;

  {
    /* Constructor using begin and end.  */
    auto r = int_array_iterator_range (begin, end);
    test_iterator_range_1 (r, array, array_size, begin, end);
  }

  {
    /* Constructor using begin, assuming end can be default-constructed.  */
    auto r2 = int_array_iterator_range (begin);
    test_iterator_range_1 (r2, array, array_size, begin, end);
  }

  {
    /* Empty range.  */
    auto r3 = int_array_iterator_range ();
    test_iterator_range_1 (r3, nullptr, 0, end, end);
  }

  {
    auto r4 = int_array_iterator_range (begin, end);

    /* Copy constructor.  */
    auto r5 (r4);
    test_iterator_range_1 (r5, array, array_size, begin, end);

    /* Move constructor.  */
    auto r6 (std::move (r4));
    test_iterator_range_1 (r6, array, array_size, begin, end);
  }

  {
    const auto r7 = int_array_iterator_range (begin, end);

    /* Const copy constructor.  */
    auto r8 (r7);
    test_iterator_range_1 (r8, array, array_size, begin, end);
  }
}

} /* namespace selftests */

INIT_GDB_FILE (iterator_range_selftests)
{
  selftests::register_test ("iterator_range", selftests::test_iterator_range);
}
