/* This testcase is part of GDB, the GNU debugger.

   Copyright 2025-2026 Free Software Foundation, Inc.

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

#ifndef SOLIB_WITH_DWZ_FOO_H
#define SOLIB_WITH_DWZ_FOO_H

extern void process_foo_data (int value);

typedef void (*foo_callback_type) (void);

extern void do_callback (foo_callback_type cb);

#endif /* SOLIB_WITH_DWZ_FOO_H */
