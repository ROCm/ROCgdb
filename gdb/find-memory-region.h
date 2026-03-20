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

#ifndef GDB_FIND_MEMORY_REGION_H
#define GDB_FIND_MEMORY_REGION_H

#include "gdbsupport/function-view.h"

/* Process memory area starting at ADDR with length SIZE.  Area is readable iff
   READ is true, writable if WRITE is true, executable if EXEC is true.  Area
   is possibly changed against its original file based copy if MODIFIED is true.

   MEMORY_TAGGED is true if the memory region contains memory tags, false
   otherwise.

   Return true on success, false otherwise.  */

using find_memory_region_ftype
  = gdb::function_view<bool (CORE_ADDR addr, unsigned long size, bool read,
			     bool write, bool exec, bool modified,
			     bool memory_tagged)>;

#endif /* GDB_FIND_MEMORY_REGION_H */
