/* Gdb/Python header exposing missing symbols in the Python limited API.
   Note: this is a workaround solution until those existing symbols below,
   or new symbols are exposed in the limited API.

   Copyright (C) 2026 Free Software Foundation, Inc.

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

#ifndef GDB_PYTHON_PYTHON_LIMITED_API_MISSING_H
#define GDB_PYTHON_PYTHON_LIMITED_API_MISSING_H

#ifdef Py_LIMITED_API
extern "C"
{

/* Symbols belonging to the configuration API introduced in PEP-741, and
   required in gdb_PyInitializer.  */

using PyInitConfig = struct PyInitConfig;

PyAPI_FUNC(PyInitConfig*) PyInitConfig_Create (void);
PyAPI_FUNC(void) PyInitConfig_Free (PyInitConfig *config);

PyAPI_FUNC(int) PyInitConfig_SetInt (PyInitConfig *config,
				     const char *name,
				     int64_t value);
PyAPI_FUNC(int) PyInitConfig_SetStr (PyInitConfig *config,
				     const char *name,
				     const char *value);

PyAPI_FUNC(int) PyInitConfig_GetError (PyInitConfig* config,
				       const char **err_msg);
PyAPI_FUNC(int) PyInitConfig_GetExitCode (PyInitConfig* config,
					  int *exitcode);

PyAPI_FUNC(int) Py_InitializeFromInitConfig (PyInitConfig *config);

/* Handler for GDB's readline support.  */

PyAPI_DATA(char) *(*PyOS_ReadlineFunctionPointer)(FILE *, FILE *, const char *);

/* Utils from Python's high level layer API.  */

PyAPI_FUNC(int) PyRun_InteractiveLoop (FILE *f, const char *p);

}
#endif /* Py_LIMITED_API */

#endif /* GDB_PYTHON_PYTHON_LIMITED_API_MISSING_H */
