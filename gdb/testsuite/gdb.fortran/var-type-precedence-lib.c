/* Copyright 2026 Free Software Foundation, Inc.

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

/* Define a struct type that will conflict with Fortran variable name.  */
struct type_shadowing_var
{
  int value;
};

/* Global variables to ensure types are in debug info.  */
static struct type_shadowing_var global_conflicting_var = {42};

/* Function to ensure library is linked and types are used.  */
void
fortran_var_type_order_test (void)
{
  global_conflicting_var.value = 42;
}
