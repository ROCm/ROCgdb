/* GDB HIP testing mode.

   Copyright (C) 2021 Free Software Foundation, Inc.
   Copyright (C) 2021 Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef GDB_HIP_TEST_MODE_H
#define GDB_HIP_TEST_MODE_H

#include "gdbsupport/gdb_string_view.h"

struct objfile;

const char *gdb_hip_test_mode_frob_names (struct objfile *objfile,
					  const char *name);
gdb::string_view gdb_hip_test_mode_frob_names (struct objfile *objfile,
					       gdb::string_view name);

#endif
