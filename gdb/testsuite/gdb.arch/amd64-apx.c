/* Copyright 2024-2026 Free Software Foundation, Inc.

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
   along with this program.  If not, see <http://www.gnu.org/licenses/>.

   Test program for Intel APX Extended GPRs (EGPRs).  */

#include <stdint.h>

int64_t data[] = {
  0x0000000004030201,
  0x0000000014131211,
  0x0000000024232221,
  0x0000000034333231,
  0x0000000044434241,
  0x0000000054535251,
  0x0000000064636261,
  0x0000000074737271,
  0x0000000084838281,
  0x0000000094939291,
  0x00000000a4a3a2a1,
  0x00000000b4b3b2b1,
  0x00000000c4c3c2c1,
  0x00000000d4d3d2d1,
  0x00000000e4e3e2e1,
  0x00000000f4f3f2f1,
};

int
main (int argc, char **argv)
{
  asm ("mov 0(%0), %%r16\n\t"
       "mov 8(%0), %%r17\n\t"
       "mov 16(%0), %%r18\n\t"
       "mov 24(%0), %%r19\n\t"
       "mov 32(%0), %%r20\n\t"
       "mov 40(%0), %%r21\n\t"
       "mov 48(%0), %%r22\n\t"
       "mov 56(%0), %%r23\n\t"
       "mov 64(%0), %%r24\n\t"
       "mov 72(%0), %%r25\n\t"
       "mov 80(%0), %%r26\n\t"
       "mov 88(%0), %%r27\n\t"
       "mov 96(%0), %%r28\n\t"
       "mov 104(%0), %%r29\n\t"
       "mov 112(%0), %%r30\n\t"
       "mov 120(%0), %%r31\n\t"
       : /* no output operands */
       : "r" (data)
       : "r16", "r17", "r18", "r19", "r20", "r21", "r22", "r23",
	 "r24", "r25", "r26", "r27", "r28", "r29", "r30", "r31");

  asm ("nop"); /* break here */

  /* Generate SIGSEGV to crash.  */
  *(volatile int *) 0;

  return 0;
}
