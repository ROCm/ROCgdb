/* This test program is part of GDB, the GNU debugger.

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

/* Simulate loading of a library JIT code.  */

#include <stdio.h>
#include <stdlib.h>

#ifdef __WIN32__
#include <windows.h>
#define dlopen(name, mode) LoadLibrary (TEXT (name))
#define dlclose(handle) FreeLibrary (handle)
#define dlerror() "an error occurred"
#else
#include <dlfcn.h>
#endif


int
main (void)
{
  void *handle = dlopen (SHLIB_NAME, RTLD_NOW);

  if (handle == NULL)
    {
      fprintf (stderr, "%s\n", dlerror ());
      exit (1);
    }

  dlclose (handle);
  return 0;
}
