/* Self tests for the enumerate range adapter.

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
#include "gdbsupport/enumerate.h"

#include <vector>
#include <array>

namespace selftests {

static void
test_enumerate ()
{
  /* Test basic enumeration over a vector.  */
  {
    std::vector<int> vec = { 10, 20, 30, 40 };
    std::vector<std::pair<std::size_t, int>> result;
    std::vector<std::pair<std::size_t, int>> expected {
      {0, 10},
      {1, 20},
      {2, 30},
      {3, 40}
    };

    for (auto [i, val] : gdb::ranges::views::enumerate (vec))
      result.push_back ({ i, val });

    SELF_CHECK (result == expected);
  }

  /* Test enumeration over an std::array.  */
  {
    std::array<int, 3> arr = { 5, 6, 7 };
    std::vector<std::pair<std::size_t, int>> result;
    std::vector<std::pair<std::size_t, int>> expected {
      {0, 5},
      {1, 6},
      {2, 7}
    };

    for (auto [i, val] : gdb::ranges::views::enumerate (arr))
      result.push_back ({ i, val });

    SELF_CHECK (result == expected);
  }

  /* Test enumeration over a C array.  */
  {
    int arr[] = { 8, 9, 10 };
    std::vector<std::pair<std::size_t, int>> result;
    std::vector<std::pair<std::size_t, int>> expected {
      {0, 8},
      {1, 9},
      {2, 10}
    };

    for (auto [i, val] : gdb::ranges::views::enumerate (arr))
      result.push_back ({ i, val });

    SELF_CHECK (result == expected);
  }

  /* Test that enumeration allows modification of elements.  */
  {
    std::vector<int> vec = { 1, 2, 3 };
    std::vector<int> expected = { 10, 20, 30 };
    std::vector<int> actual_i;
    std::vector<int> expected_i = { 0, 1, 2 };

    for (auto [i, val] : gdb::ranges::views::enumerate (vec))
      {
	val *= 10;
	actual_i.push_back (i);
      }

    SELF_CHECK (vec == expected);
    SELF_CHECK (actual_i == expected_i);
  }

  /* Test enumeration over an empty container.  */
  {
    std::vector<int> vec;
    std::vector<std::pair<std::size_t, int>> result;
    std::vector<std::pair<std::size_t, int>> expected;

    for (auto [i, val] : gdb::ranges::views::enumerate (vec))
      result.push_back ({ i, val });

    SELF_CHECK (result == expected);
  }

  /* Test enumeration over a single-element container.  */
  {
    std::vector<int> vec = { 42 };
    std::vector<std::pair<std::size_t, int>> result;
    std::vector<std::pair<std::size_t, int>> expected {
      {0, 42}
    };

    for (auto [i, val] : gdb::ranges::views::enumerate (vec))
      result.push_back ({ i, val });

    SELF_CHECK (result == expected);
  }

  /* Test enumeration over an rvalue container.  */
  {
    std::vector<std::pair<std::size_t, int>> result;
    std::vector<std::pair<std::size_t, int>> expected {
      {0, 17},
      {1, 38},
      {2, 99}
    };

    for (auto [i, val] :
	 gdb::ranges::views::enumerate (std::vector<int> { 17, 38, 99 }))
      result.push_back ({ i, val });

    SELF_CHECK (result == expected);
  }

  /* Test enumeration with const container.  */
  {
    const std::vector<int> vec = { 100, 200 };
    std::vector<std::pair<std::size_t, int>> result;
    std::vector<std::pair<std::size_t, int>> expected {
      {0, 100},
      {1, 200}
    };

    for (auto [i, val] : gdb::ranges::views::enumerate (vec))
      result.push_back ({ i, val });

    SELF_CHECK (result == expected);
  }
}

} /* namespace selftests */

INIT_GDB_FILE (enumerate_selftests)
{
  selftests::register_test ("enumerate", selftests::test_enumerate);
}
