/* This testcase is part of GDB, the GNU debugger.

   Copyright 2026 Free Software Foundation, Inc.

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

#ifndef SOURCE_TRACKING_INLINE_HDR_H
#define SOURCE_TRACKING_INLINE_HDR_H

static inline void __attribute__ ((__always_inline__))
header_func (void)
{
  static int header_var = 0;

  header_var = header_var + 1;
  header_var = header_var + 2;		/* Header breakpoint location.  */
  header_var = header_var + 3;
}

extern int other_func (int arg);

#endif /* SOURCE_TRACKING_INLINE_HDR_H */
