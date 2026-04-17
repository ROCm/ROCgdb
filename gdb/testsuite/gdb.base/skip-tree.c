/* This testcase is part of GDB, the GNU debugger.

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

extern void func1 (void);
extern void func2 (void);
extern void func3 (void);
extern void func4 (void);

static volatile int global_var;

void
func5 (void)
{
  /* Nothing.  */
}

int
main (void)
{
  /* Some filler before the real work starts.  */
  ++global_var;

  func1 ();	/* In aa/bb/file.c */
  func2 ();	/* In cc/bb/file.c */
  func3 ();	/* In dd/ee/file.c */
  func4 ();	/* In dd/ee/ff/file.c */
  func5 ();	/* Makes exp file simpler.  */

  /* Some filler before the function ends.  */
  ++global_var;

  return 0;
}
