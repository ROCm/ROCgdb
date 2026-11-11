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

#include <algorithm>

/* The 'equal' wrapper here only works on spans, so include this.
   When moving to C++20, this will be removed -- in fact this entire
   file should be removed.  */
#include "gdbsupport/span.h"

namespace gdb {
namespace ranges {

/* Compare LHS and RHS for (deep) equality.  That is, whether LHS and
   RHS have the same sizes, and whether each pair of elements of LHS
   and RHS at the same position compares equal.  */

template <typename T>
bool
equal (const gdb::span<T> &lhs, const gdb::span<T> &rhs)
{
#if __cplusplus >= 202002L
  return std::ranges::equal (lhs, rhs);
#else
  if (lhs.size () != rhs.size ())
    return false;

  for (size_t i = 0; i < lhs.size (); i++)
    if (!(lhs[i] == rhs[i]))
      return false;

  return true;
#endif
}

/* Replace all occurrences of a value in the provided range.

   Note: this helper is a reimplementation of std::ranges::replace, only
   available from C++20 onwards, and consequently, should be replaced by
   std::ranges::replace once GDB switches to C++20.  */

template <class Range, typename T>
void replace (Range &&r, const T &old_value, const T &new_value)
{
#if __cplusplus >= 202002L
  std::ranges::replace (r, old_value, new_value);
#else
  std::replace (r.begin (), r.end (), old_value, new_value);
#endif
}

} /* namespace ranges */
} /* namespace gdb */

#endif /* GDBSUPPORT_RANGES_H */
