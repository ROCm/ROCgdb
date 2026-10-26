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

#include <stddef.h>

extern void *dummy_malloc (size_t size);

typedef void *(*malloc_t) (size_t size);

#ifndef IFUNC_RESOLVER_ATTR
asm (".type malloc, %gnu_indirect_function");
malloc_t
malloc (unsigned long hwcap)
#else
static malloc_t
resolve_malloc (void)
#endif
{
#ifndef IFUNC_RESOLVER_ATTR
  (void) hwcap;
#endif
  return dummy_malloc;
}

#ifdef IFUNC_RESOLVER_ATTR
extern void *malloc (size_t size);

__typeof (malloc) malloc __attribute__ ((ifunc ("resolve_malloc")));
#endif
