/* Copyright (C) 2026 Free Software Foundation, Inc.

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

#ifndef GDBSUPPORT_RANGES_H
#define GDBSUPPORT_RANGES_H

namespace gdb {
namespace ranges {

/* Replace all occurrences of a value in the provided range.

   Note: this helper is a reimplementation of std::ranges::replace, only
   available from C++20 onwards, and consequently, should be replaced by
   std::ranges::replace once GDB switches to C++20.  */

template <class Range, typename T>
void replace (Range &&r, const T &old_value, const T &new_value)
{
  std::replace (r.begin (), r.end (), old_value, new_value);
}

} /* namespace ranges */
} /* namespace gdb */

#endif /* GDBSUPPORT_RANGES_H */
