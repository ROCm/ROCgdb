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

/* Exercise AArch64's POE Extension.  */

/* This test is based on the Linux kernel documentation for Memory
   Protection Keys, including the arm64 Permission Overlay Extension
   (FEAT_S1POE) support described in
   Documentation/core-api/protection-keys.rst.  */

#define _GNU_SOURCE
#include <sys/mman.h>
#include <unistd.h>

int
main (void)
{
  long pagesize = sysconf (_SC_PAGESIZE);

  int *buf = mmap (NULL, pagesize, PROT_READ | PROT_WRITE,
		   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (buf == MAP_FAILED)
    return 1;

  int pkey = pkey_alloc (0, 0);
  if (pkey == -1)
    return 1;

  if (pkey_mprotect (buf, pagesize, PROT_READ | PROT_WRITE, pkey) == -1)
    return 1;

  if (pkey_set (pkey, PKEY_DISABLE_WRITE) == -1)
    return 1;

  buf[0] = 42;  /* Expect SIGSEGV POE violation.  */

  return 0;
}
