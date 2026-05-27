/* This test program is part of GDB, the GNU debugger.

   Copyright 2024-2026 Free Software Foundation, Inc.

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

#include <stdint.h>

#define PREPARE_REGS(X, Y, Z) \
	__asm__ volatile ("mov x19, %0\n" : : "r"(X): "x19"); \
	__asm__ volatile ("mov x20, %0\n" : : "r"(Y): "x20"); \
	__asm__ volatile ("mov x21, %0\n" : : "r"(Z): "x21");

int
main (void)
{
  const uint64_t a = 0x0123456789abcdef;
  const uint64_t b = 0xfedbca9876543210;
  const uint64_t c = 0xdeadbeefc001face;

  PREPARE_REGS (a, b, c);
  /* Before abs.  */
  __asm__ volatile ("abs x19, x21\n" ::: "x19");
  /* After abs.  */

  PREPARE_REGS (a, b, c);
  /* Before cnt.  */
  __asm__ volatile ("cnt x19, x20\n" : :: "x19");
  /* After cnt.  */

  PREPARE_REGS (a, b, c);
  /* Before ctz.  */
  __asm__ volatile ("ctz x19, x20\n" : :: "x19");
  /* After ctz.  */

  PREPARE_REGS (a, b, c);
  /* Before smax-1.  */
  __asm__ volatile ("smax x19, x20, #10\n" : :: "x19");
  /* After smax-1.  */

  PREPARE_REGS (a, b, c);
  /* Before smax-2.  */
  __asm__ volatile ("smax x19, x20, x21\n" : :: "x19");
  /* After smax-2.  */

  PREPARE_REGS (a, b, c);
  /* Before smin-1.  */
  __asm__ volatile ("smin x19, x20, #10\n" : :: "x19");
  /* After smin-1.  */

  PREPARE_REGS (a, b, c);
  /* Before smin-2.  */
  __asm__ volatile ("smin x19, x20, x21\n" : :: "x19");
  /* After smin-2.  */

  PREPARE_REGS (a, b, c);
  /* Before umax-1.  */
  __asm__ volatile ("umax x19, x20, #10\n" : :: "x19");
  /* After umax-1.  */

  PREPARE_REGS (a, b, c);
  /* Before umax-2.  */
  __asm__ volatile ("umax x19, x20, x21\n" : :: "x19");
  /* After umax-2.  */

  PREPARE_REGS (a, b, c);
  /* Before umin-1.  */
  __asm__ volatile ("umin x19, x20, #10\n" : :: "x19");
  /* After umin-1.  */

  PREPARE_REGS (a, b, c);
  /* Before umin-2.  */
  __asm__ volatile ("umin x19, x20, x21\n" : :: "x19");
  /* After umin-2.  */

  return 0;
}
