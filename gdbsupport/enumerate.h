/* An enumerate range adapter for GDB, the GNU debugger.
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

#ifndef GDBSUPPORT_ENUMERATE_H
#define GDBSUPPORT_ENUMERATE_H

#include <cstddef>
#include <iterator>
#include <tuple>
#include <utility>

namespace gdb::ranges::views
{

/* An iterator that wraps another iterator and yields tuples containing
   both the index and the value.  */

template<typename Iterator>
class enumerate_iterator
{
  using base_iterator = Iterator;
public:
  using value_type
    = std::tuple<std::size_t,
		 typename std::iterator_traits<base_iterator>::reference>;

  explicit enumerate_iterator (Iterator it)
    : m_it (std::move (it))
  {}

  value_type operator* () const
  { return { m_index, *m_it }; }

  enumerate_iterator &operator++ ()
  {
    ++m_it;
    ++m_index;
    return *this;
  }

  bool operator== (const enumerate_iterator &other) const
  { return m_it == other.m_it; }

  bool operator!= (const enumerate_iterator &other) const
  { return m_it != other.m_it; }

private:
  Iterator m_it;
  std::size_t m_index = 0;
};

/* A range adapter that allows iteration on both index and value.  */

template<typename Range>
class enumerate_range
{
  using base_iterator = decltype (std::begin (std::declval<Range &> ()));

public:
  using iterator = enumerate_iterator<base_iterator>;

  explicit enumerate_range (Range &&range)
    : m_range (std::forward<Range> (range))
  {}

  iterator begin ()
  { return iterator (std::begin (m_range)); }

  iterator end ()
  { return iterator (std::end (m_range)); }

private:
  Range m_range;
};

/* Return an enumerate_range for RANGE, allowing iteration with both
   index and value.

   Example usage:

     std::vector<int> vec = {10, 20, 30};
     for (auto [i, val] : gdb::ranges::views::enumerate (vec))
       printf ("%zu: %d\n", i, val);

   This prints:

     0: 10
     1: 20
     2: 30

   The value is a reference to the element in the container, so
   modifications are possible:

     for (auto [i, val] : gdb::ranges::views::enumerate (vec))
       val *= 2;  */

template<typename Range>
enumerate_range<Range>
enumerate (Range &&range)
{
  return enumerate_range<Range> (std::forward<Range> (range));
}

} /* namespace gdb::ranges::views */

#endif /* GDBSUPPORT_ENUMERATE_H */
