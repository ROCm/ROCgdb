/* Host dependent utilities used by the amd-dbgapi target.

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

#ifndef GDB_AMD_DBGAPI_HDEP_H
#define GDB_AMD_DBGAPI_HDEP_H

#include <amd-dbgapi/amd-dbgapi.h>

/* Null amd_dbgapi_notifier_t.  */
extern const amd_dbgapi_notifier_t null_amd_dbgapi_notifier;

/* Clear the notifier.  */
extern void amd_dbgapi_notifier_clear (amd_dbgapi_notifier_t notifier);

/* Get the file descriptor associated with the notifier.  */
extern int amd_dbgapi_notifier_get_fd (amd_dbgapi_notifier_t notifier);

/* Ensure that we do not keep a reference to the notifier NOTIFIER which
   is about to get invalidated.  */
extern void amd_dbgapi_notifier_release (amd_dbgapi_notifier_t notifier);

/* Get the amd-dbgapi shared library file path.  */
extern const char *get_dbgapi_library_file_path ();

#endif /* GDB_AMD_DBGAPI_HDEP_H  */
