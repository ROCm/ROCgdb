/* Copyright (C) 2024-2026 Free Software Foundation, Inc.

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

#ifndef GDBSUPPORT_UNORDERED_MAP_H
#define GDBSUPPORT_UNORDERED_MAP_H

#include "unordered_dense/unordered_dense.h"

namespace gdb
{

template<typename Key,
	 typename T,
	 typename Hash = ankerl::unordered_dense::hash<Key>,
	 typename KeyEqual = std::equal_to<Key>>
using unordered_map
  = ankerl::unordered_dense::map
      <Key, T, Hash, KeyEqual, std::allocator<std::pair<Key, T>>,
       ankerl::unordered_dense::bucket_type::standard>;

/* An unordered_map with std::string keys that supports transparent
   lookup from std::string_view, avoiding the construction of temporary
   std::string objects during lookups.  std::string_view is implicitly
   constructible from `const char *` and `std::string`, so it covers those
   too.  */

namespace detail
{

struct unordered_string_map_hash
{
  using is_transparent = void;
  using is_avalanching = void;

  std::uint64_t operator() (std::string_view sv) const noexcept
  { return ankerl::unordered_dense::hash<std::string_view> () (sv); }
};

struct unordered_string_map_eq
{
  using is_transparent = void;

  bool operator() (std::string_view lhs, std::string_view rhs) const noexcept
  { return lhs == rhs; }
};

} /* namespace detail */

template<typename T>
using unordered_string_map
  = gdb::unordered_map<std::string, T,
		       detail::unordered_string_map_hash,
		       detail::unordered_string_map_eq>;

} /* namespace gdb */

#endif /* GDBSUPPORT_UNORDERED_MAP_H */
