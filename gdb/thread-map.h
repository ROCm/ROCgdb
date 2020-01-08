/* Thread map type for GDB, the GNU debugger.
   Copyright (C) 2019 Free Software Foundation, Inc.

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

#ifndef THREAD_MAP_H
#define THREAD_MAP_H

#include "defs.h"

#include <unordered_map>

struct thread_info;

using ptid_thread_map = std::unordered_map<ptid_t, thread_info *, hash_ptid>;

struct all_thread_map_range_iterator
{
  all_thread_map_range_iterator (ptid_thread_map::const_iterator iter)
    : m_iter (iter)
  {}

  bool operator!= (const all_thread_map_range_iterator &other)
  { return this->m_iter != other.m_iter; }

  void operator++ ()
  { this->m_iter++; }

  thread_info *operator* ()
  { return this->m_iter->second; }

private:
  typename ptid_thread_map::const_iterator m_iter;
};

struct all_thread_map_range
{
  all_thread_map_range (const ptid_thread_map &map)
    : m_map (map)
  {}

  all_thread_map_range_iterator begin ()
  {
    return all_thread_map_range_iterator (this->m_map.begin ());
  }

  all_thread_map_range_iterator end ()
  {
    return all_thread_map_range_iterator (this->m_map.end ());
  }

private:
  const ptid_thread_map &m_map;
};

struct non_exited_thread_map_range_iterator
{
  non_exited_thread_map_range_iterator (typename ptid_thread_map::const_iterator iter,
					typename ptid_thread_map::const_iterator end)
    : m_iter (iter), m_end (end)
  {
    advante_to_next_matching ();
  }

  bool operator!= (const non_exited_thread_map_range_iterator &other)
  { return this->m_iter != other.m_iter; }

  void operator++ ()
  {
    this->m_iter++;
    advante_to_next_matching ();
  }

  thread_info *operator* ()
  { return this->m_iter->second; }

private:
  typename ptid_thread_map::const_iterator m_iter;
  typename ptid_thread_map::const_iterator m_end;

  void advante_to_next_matching ()
  {
    while (this->m_iter != this->m_end
	   && this->m_iter->second->state == THREAD_EXITED)
      {
	this->m_iter++;
      }
  }
};

struct non_exited_thread_map_range
{
  non_exited_thread_map_range (const ptid_thread_map &map)
    : m_map (map)
  {}

  non_exited_thread_map_range_iterator begin()
  {
    return non_exited_thread_map_range_iterator (this->m_map.begin (), this->m_map.end ());
  }

  non_exited_thread_map_range_iterator end()
  {
    return non_exited_thread_map_range_iterator (this->m_map.end (), this->m_map.end ());
  }

private:
  const ptid_thread_map &m_map;
};

#endif
