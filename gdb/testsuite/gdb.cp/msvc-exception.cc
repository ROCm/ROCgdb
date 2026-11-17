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

/* Exercise catch throw, catch rethrow, catch catch and next over a
   throw with C++ exceptions on Windows.  The program is built by the
   test with whatever C++ compiler and runtime the board provides.  */

#include <stdexcept>
#include <cstdio>

int testval = -1;

/* Throws std::runtime_error.  */
int
foo (int i)
{
  if (i < 32)
    throw std::runtime_error ("boom");   /* throw-site */
  else
    return i * 2;
}

/* Throws and catches locally (used for next-over-throw test).  */
void
local_catch (int val)
{
  try
    {
      throw std::runtime_error ("local");
    }
  catch (const std::exception &)
    {
      testval = val;
    }
}

/* Called when we catch, so we have a known symbol.  */
int
catcher (int x)
{
  return x;
}

/* Rethrows.  */
void
rethrower ()
{
  try
    {
      throw std::runtime_error ("rethrow");
    }
  catch (...)
    {
      throw;
    }
}

int
main ()
{
  int j;

  /* Test 1: basic throw and catch.  */
  try
    {
      j = foo (20);          /* basic-throw */
    }
  catch (const std::exception &x)
    {
      catcher (13);          /* basic-catch */
    }

  /* Test 2: nested try/catch with rethrow.  */
  try
    {
      try
	{
	  j = foo (20);      /* nested-throw */
	}
      catch (const std::exception &x)
	{
	  catcher (13);
	  throw;             /* rethrow-site */
	}
    }
  catch (const std::exception &y)
    {
      catcher (13);          /* outer-catch */
    }

  /* Test 3: next-over-throw scenarios.  */
  local_catch (1);           /* step-target-s1 */
  printf ("after s1: %d\n", testval);   /* after-s1 */

  try                        /* step-target-s2 */
    {
      rethrower ();
    }
  catch (const std::exception &)  /* s2-catch-clause */
    {
      testval = 2;
    }
  printf ("after s2: %d\n", testval);   /* after-s2 */

  printf ("done testval=%d\n", testval);
  return 0;
}
