/* Posix implementation of the host dependent utilities for the amd-dbgapi
   target.

   Copyright (C) 2021-2024 Free Software Foundation, Inc.
   Copyright (C) 2021-2024 Advanced Micro Devices, Inc. All rights reserved.

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

#include "amd-dbgapi-hdep.h"
#include <amd-dbgapi/amd-dbgapi.h>
#include <dlfcn.h>

/* See amd-dbgapi-hdep.h.  */
const char *
get_dbgapi_library_file_path ()
{
  Dl_info dl_info{};
  if (!dladdr ((void *) amd_dbgapi_get_version, &dl_info))
    return "";
  return dl_info.dli_fname;
}
