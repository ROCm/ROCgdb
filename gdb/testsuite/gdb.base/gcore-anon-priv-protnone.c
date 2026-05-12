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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

void
done ()
{
}

void *mmapped_data;

int
main ()
{
  /* 10 pages on most systems.  */
  size_t sz = 4096 * 10;

  /* Allocate anonymous memory.  */
  mmapped_data = mmap (NULL, sz, PROT_READ | PROT_WRITE,
		     MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (mmapped_data == MAP_FAILED)
    {
      perror ("mmap");
      return 1;
    }

#ifdef FILL_WITH_DATA
  /* Fill with a recognizable pattern.  */
  memset (mmapped_data, 0xab, sz);
#endif

  /* Remove all permissions.  */
  if (mprotect (mmapped_data, sz, PROT_NONE) == -1)
    {
      perror ("mprotect");
      return 1;
    }

  done ();
  return 0;
}
