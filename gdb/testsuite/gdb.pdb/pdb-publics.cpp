/* Copyright 2026 Free Software Foundation, Inc.

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

/* The publics of a function, of a C++ global whose public has a decorated
   name, and of a function called through an import thunk.  */

/* Declared without __declspec(dllimport), so that the call goes through
   the import thunk the linker makes for it.  */
extern "C" unsigned long GetCurrentProcessId (void);

int g_counter = 0;

int
main (void)
{
  g_counter++;
  return (int) (GetCurrentProcessId () & 1);
}
