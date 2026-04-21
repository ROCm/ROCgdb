/* Self tests for the filtered_iterator class.

   Copyright (C) 2019-2026 Free Software Foundation, Inc.

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
#include "gdbsupport/filtered-iterator.h"
#include "int-array-iterator.h"

namespace selftests {

/* Filter to only keep the even numbers.  */

struct even_numbers_only
{
  bool operator() (int n)
  {
    return n % 2 == 0;
  }
};

/* Test typical usage.  */

static void
test_filtered_iterator ()
{
  int array[] = { 4, 4, 5, 6, 7, 8, 9 };
  std::vector<int> even_ints;
  const std::vector<int> expected_even_ints { 4, 4, 6, 8 };

  int_array_iterator begin (array, ARRAY_SIZE (array));
  int_array_iterator end;
  filtered_iterator<int_array_iterator, even_numbers_only>
    filtered_iter (begin, end);
  filtered_iterator<int_array_iterator, even_numbers_only>
    filtered_end (end, end);

  for (; filtered_iter != filtered_end; ++filtered_iter)
    even_ints.push_back (*filtered_iter);

  SELF_CHECK (even_ints == expected_even_ints);
}

/* Same as the above, but using pointers as the iterator base type.  */

static void
test_filtered_iterator_ptr ()
{
  int array[] = { 4, 4, 5, 6, 7, 8, 9 };
  std::vector<int> even_ints;
  const std::vector<int> expected_even_ints { 4, 4, 6, 8 };

  filtered_iterator<int *, even_numbers_only> iter
    (array, array + ARRAY_SIZE (array));
  filtered_iterator<int *, even_numbers_only> end
    (array + ARRAY_SIZE (array), array + ARRAY_SIZE (array));

  for (; iter != end; ++iter)
    even_ints.push_back (*iter);

  SELF_CHECK (even_ints == expected_even_ints);
}

/* Test operator== and operator!=. */

static void
test_filtered_iterator_eq ()
{
  int array[] = { 4, 4, 5, 6, 7, 8, 9 };

  int_array_iterator begin (array, ARRAY_SIZE (array));
  int_array_iterator end;
  filtered_iterator<int_array_iterator, even_numbers_only>
    iter1 (begin, end);
  filtered_iterator<int_array_iterator, even_numbers_only>
    iter2 (begin, end);

  /* They start equal.  */
  SELF_CHECK (iter1 == iter2);
  SELF_CHECK (!(iter1 != iter2));

  /* Advance 1, now they aren't equal (despite pointing to equal values).  */
  ++iter1;
  SELF_CHECK (!(iter1 == iter2));
  SELF_CHECK (iter1 != iter2);

  /* Advance 2, now they are equal again.  */
  ++iter2;
  SELF_CHECK (iter1 == iter2);
  SELF_CHECK (!(iter1 != iter2));
}


/* Same as the above, but using pointers as the iterator base type.  */

static void
test_filtered_iterator_eq_ptr ()
{
  int array[] = { 4, 4, 5, 6, 7, 8, 9 };

  filtered_iterator<int *, even_numbers_only> iter1
    (array, array + ARRAY_SIZE(array));
  filtered_iterator<int *, even_numbers_only> iter2
    (array, array + ARRAY_SIZE(array));

  /* They start equal.  */
  SELF_CHECK (iter1 == iter2);
  SELF_CHECK (!(iter1 != iter2));

  /* Advance 1, now they aren't equal (despite pointing to equal values).  */
  ++iter1;
  SELF_CHECK (!(iter1 == iter2));
  SELF_CHECK (iter1 != iter2);

  /* Advance 2, now they are equal again.  */
  ++iter2;
  SELF_CHECK (iter1 == iter2);
  SELF_CHECK (!(iter1 != iter2));
}

} /* namespace selftests */

INIT_GDB_FILE (filtered_iterator_selftests)
{
  selftests::register_test ("filtered_iterator",
			    selftests::test_filtered_iterator);
  selftests::register_test ("filtered_iterator_eq",
			    selftests::test_filtered_iterator_eq);
  selftests::register_test ("filtered_iterator_ptr",
			    selftests::test_filtered_iterator_ptr);
  selftests::register_test ("filtered_iterator_eq_ptr",
			    selftests::test_filtered_iterator_eq_ptr);
}
