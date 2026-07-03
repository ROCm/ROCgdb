/* Self tests for gdb::ranges algorithms for GDB, the GNU debugger.

   Copyright (C) 2017-2026 Free Software Foundation, Inc.

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
#include "gdbsupport/array-view.h"
#include "gdbsupport/ranges.h"
#include <array>
#include <vector>

namespace selftests {
namespace ranges_replace_tests {

/* Entry point.  */

static void
run_tests ()
{
  /* With owning container, no copy and elements correctly replaced.  */
  {
    std::vector<int> vec = {1, 2, 3, 2};
    gdb::ranges::replace (vec, 2, 1);
    SELF_CHECK (vec[0] == 1);
    SELF_CHECK (vec[1] == 1);
    SELF_CHECK (vec[2] == 3);
    SELF_CHECK (vec[3] == 1);
  }

  /* With non-owning container.  */
  {
    std::array<int, 4> array = {1, 2, 3, 2};
    gdb::array_view<int> view = array;
    gdb::ranges::replace (view, 2, 1);
    SELF_CHECK (array[0] == 1);
    SELF_CHECK (array[1] == 1);
    SELF_CHECK (array[2] == 3);
    SELF_CHECK (array[3] == 1);
  }
}

} /* namespace ranges_replaceq_tests */
} /* namespace selftests */

INIT_GDB_FILE (ranges_selftests)
{
  selftests::register_test ("ranges_replace",
			    selftests::ranges_replace_tests::run_tests);
}
