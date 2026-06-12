/* Self tests for the next_iterator class.

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
#include "gdbsupport/next-iterator.h"
#include <vector>

namespace selftests {
namespace next_iterator {

struct list;
using list_iterator = ::next_iterator<struct list>;
using list_range = next_range<struct list>;

/* The next_iterator allows usage with an incomplete type, provided the type
   is defined later in the same file.  While not essential to its functioning,
   we're currently using this property in the sources, so check/demonstrate
   it.  */

static void
test_next_iterator_incomplete_type (struct list *l,
				    std::vector<int> &expected)
{
  list_iterator begin (l);
  list_iterator end;

  unsigned cnt = 0;
  list_iterator i;
  for (i = begin; i != end; ++i)
    cnt++;
  SELF_CHECK (cnt == expected.size ());
}

struct list
{
  int i;
  struct list *next;
};

static void
test_next_iterator ()
{
  struct list a = { 1, nullptr };
  struct list b = { 2, &a };
  struct list c = { 3, &b };

  /* Constructor with parameter.  */
  list_iterator begin (&c);

  /* Constructor without parameter.  */
  list_iterator end;

  /* Operator*.  */
  SELF_CHECK (*begin == &c);
  SELF_CHECK (*end == nullptr);

  /* Operator==, operator!=.  */
  SELF_CHECK (begin == begin);
  SELF_CHECK (end == end);
  SELF_CHECK (begin != end);

  /* Operator++.  */
  list_iterator i (&c);
  ++i;
  SELF_CHECK (*i == &b);

  /* Loop using iterators.  */
  std::vector<int> expected = {3, 2, 1};
  std::vector<int> v;
  for (i = begin; i != end; ++i)
    v.push_back ((*i)->i);
  SELF_CHECK (v == expected);

  /* Loop using range.  */
  v.clear ();
  for (auto l : list_range (begin))
    v.push_back (l->i);
  SELF_CHECK (v == expected);

  test_next_iterator_incomplete_type (&c, expected);
}

} /* namespace next_iterator */
} /* namespace selftests */

INIT_GDB_FILE (next_iterator_selftests)
{
  selftests::register_test ("next-iterator",
			    selftests::next_iterator::test_next_iterator);
}
