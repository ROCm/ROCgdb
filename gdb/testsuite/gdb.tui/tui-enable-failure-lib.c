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

/* Preload library that overrides ncurses newterm to always return NULL,
   causing tui_enable to fail.  */

#include <stddef.h>
#include <stdio.h>

/* Define this type so we can override newterm.  We only plan to
   return NULL, so the details of this type are not important.  */
typedef struct screen_dummy SCREEN;

SCREEN *
newterm (const char *type, FILE *outfd, FILE *infd)
{
  return NULL;
}
