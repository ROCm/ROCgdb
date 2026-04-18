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

#ifndef GDBSUPPORT_ITERATION_STATUS_H
#define GDBSUPPORT_ITERATION_STATUS_H

/* Return type for iteration callbacks.

   Using an enum makes it clear at each call site whether the callback wants to
   stop or continue, unlike a plain bool where the convention can change between
   APIs.  */

enum class iteration_status
{
  keep_going,
  stop,
};

#endif /* GDBSUPPORT_ITERATION_STATUS_H */
