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

#include <stdalign.h>
#include <stdint.h>
#include <string.h>

#define INITIAL_STRING "This is just some string."
#define BUF_SIZE sizeof (INITIAL_STRING)

#define PREPARE_REGS(X, Y)						    \
  strcpy (src, INITIAL_STRING);					       \
  __asm__ volatile ("mov x21, %0\n" ::"r"((uint64_t *)src) : "x21");	  \
  __asm__ volatile ("mov x19, %0\n" ::"r"(X) : "x19");			\
  __asm__ volatile ("mov x20, %0\n" ::"r"(Y) : "x20");

int
main (void)
{
  alignas (16) char src[BUF_SIZE];

  uint64_t a = 0x0123456789abcdef;
  uint64_t b = 0xfedbca9876543210;

  PREPARE_REGS (a, b);
  /* Before ldclrp.  */
  __asm__ volatile ("ldclrp x19, x20, [x21]\n" : : : "x19", "x20", "memory");
  /* After ldclrp.  */

  PREPARE_REGS (a, b);
  /* Before ldclrpa.  */
  __asm__ volatile ("ldclrpa x19, x20, [x21]\n" : : : "x19", "x20", "memory");
  /* After ldclrpa.  */

  PREPARE_REGS (a, b);
  /* Before ldclrpal.  */
  __asm__ volatile ("ldclrpal x19, x20, [x21]\n" : : : "x19", "x20", "memory");
  /* After ldclrpal.  */

  PREPARE_REGS (a, b);
  /* Before ldclrpl.  */
  __asm__ volatile ("ldclrpl x19, x20, [x21]\n" : : : "x19", "x20", "memory");
  /* After ldclrpl.  */

  PREPARE_REGS (a, b);
  /* Before ldsetp.  */
  __asm__ volatile ("ldsetp x19, x20, [x21]\n" : : : "x19", "x20", "memory");
  /* After ldsetp.  */

  PREPARE_REGS (a, b);
  /* Before ldsetpa.  */
  __asm__ volatile ("ldsetpa x19, x20, [x21]\n" : : : "x19", "x20", "memory");
  /* After ldsetpa.  */

  PREPARE_REGS (a, b);
  /* Before ldsetpal.  */
  __asm__ volatile ("ldsetpal x19, x20, [x21]\n" : : : "x19", "x20", "memory");
  /* After ldsetpal.  */

  PREPARE_REGS (a, b);
  /* Before ldsetpl.  */
  __asm__ volatile ("ldsetpl x19, x20, [x21]\n" : : : "x19", "x20", "memory");
  /* After ldsetpl.  */

  PREPARE_REGS (a, b);
  /* Before swpp.  */
  __asm__ volatile ("swpp x19, x20, [x21]\n" : : : "x19", "x20", "memory");
  /* After swpp.  */

  PREPARE_REGS (a, b);
  /* Before swppa.  */
  __asm__ volatile ("swppa x19, x20, [x21]\n" : : : "x19", "x20", "memory");
  /* After swppa.  */

  PREPARE_REGS (a, b);
  /* Before swppal.  */
  __asm__ volatile ("swppal x19, x20, [x21]\n" : : : "x19", "x20", "memory");
  /* After swppal.  */

  PREPARE_REGS (a, b);
  /* Before swppl.  */
  __asm__ volatile ("swppl x19, x20, [x21]\n" : : : "x19", "x20", "memory");
  /* After swppl.  */

  return 0;
}
