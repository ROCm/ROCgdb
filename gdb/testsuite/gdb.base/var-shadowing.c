/* Copyright (C) 2023-2026 Free Software Foundation, Inc.

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

#include <stdlib.h>

int
shadowing (void)
{
  int a = 100;  /* bp for entry */
  unsigned int val1 = 1;		/* val1-d1 */
  unsigned int val2 = 2;		/* val2-d1 */
  a = 101;  /* bp for locals 1 */
  {
    unsigned int val2 = 3;		/* val2-d2 */
    unsigned int val3 = 4;		/* val3-d1 */
    a = 102;  /* bp for locals 2 */
    {
      unsigned int val1 = 5;		/* val1-d2 */
      a = 103;  /* bp for locals 3 */
      {
	#include "var-shadowing2.c"
	unsigned int val1 = 6;	/* val1-d3 */
	unsigned int val2 = 7;	/* val2-d3 */
	unsigned int val3 = 8;	/* val3-d2 */
	a = 104;  /* bp for locals 4 */
      }
    }
  }
  a = 105;

  return 0; /* bp for locals 5 */
}

int
main (void)
{
  shadowing ();
  return 0;
}
