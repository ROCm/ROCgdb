/* Location scope support.

   Copyright (C) 2023 Free Software Foundation, Inc.
   Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef LOCATION_SCOPE_H
#define LOCATION_SCOPE_H

/* The scope of a location.  */
enum location_scopes : unsigned
{
  /* The location needs inferior information.  */
  LOCATION_SCOPE_INFERIOR = 0x1,

  /* The location description needs inferior and
     thread information.  */
  LOCATION_SCOPE_THREAD = (1 << 1) | LOCATION_SCOPE_INFERIOR,

  /* The location description needs inferior, thread and lane
     information.  */
  LOCATION_SCOPE_LANE = (1 << 2) | LOCATION_SCOPE_THREAD,

  /* The location description need inferior, thread
     and frame information.  */
  LOCATION_SCOPE_FRAME = (1 << 3) | LOCATION_SCOPE_THREAD,
};

DEF_ENUM_FLAGS_TYPE (enum location_scopes, location_scope);

/* Check whether SCOPE matches WHICH.  */
static inline bool scope_matches (location_scope scope, location_scope which)
{
  return ((scope & which) == which);
}

#endif /* LOCATION_SCOPE_H */

