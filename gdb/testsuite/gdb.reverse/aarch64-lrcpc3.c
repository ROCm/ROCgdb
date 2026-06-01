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

#define PREPARE_SRC_AND_PTR(OFFSET)					   \
  strcpy (src, INITIAL_STRING);					       \
  __asm__ volatile ("mov x21, %0"					     \
		    :							 \
		    : "r"((uint64_t *)((uint8_t *)src + (OFFSET)))	    \
		    : "x21")

#define PREPARE_GPR(OFFSET)					      \
  __asm__ volatile ("mov x19, #0\n" ::: "x19");			       \
  __asm__ volatile ("mov x20, #0\n" ::: "x20");			       \
  PREPARE_SRC_AND_PTR (OFFSET)

#define PREPARE_VECTOR_REG(OFFSET)					    \
  __asm__ volatile ("movi v22.2d, #0\n" ::: "v22");			   \
  PREPARE_SRC_AND_PTR (OFFSET)

int
main (void)
{
  alignas (16) char src[BUF_SIZE];

  PREPARE_GPR (0);
  /* Before ldiapp-0.  */
  __asm__ volatile ("ldiapp x19, x20, [x21]\n" : : : "x19", "x20", "memory");
  /* After ldiapp-0.  */

  PREPARE_GPR (0);
  /* Before ldiapp-1.  */
  __asm__ volatile ("ldiapp w19, w20, [x21]\n" : : : "x19", "x20", "memory");
  /* After ldiapp-1.  */

  PREPARE_GPR (0);
  /* Before ldiapp-2.  */
  __asm__ volatile ("ldiapp x19, x20, [x21], #16\n"
		    :
		    :
		    : "x19", "x20", "x21", "memory");
  /* After ldiapp-2.  */

  PREPARE_GPR (0);
  /* Before ldiapp-3.  */
  __asm__ volatile ("ldiapp w19, w20, [x21], #8\n"
		    :
		    :
		    : "x19", "x20", "x21", "memory");
  /* After ldiapp-3.  */
  /* Register overlap between source and destination registers.  Since there is
     no offset, writeback is disabled.  */

  PREPARE_GPR (0);
  /* Before ldiapp-4.  */
  __asm__ volatile ("ldiapp x21, x20, [x21]\n" : : : "x20", "x21", "memory");
  /* After ldiapp-4.  */

  PREPARE_GPR (0);
  /* Before ldiapp-5.  */
  __asm__ volatile ("ldiapp w21, w20, [x21]\n" : : : "x20", "x21", "memory");
  /* After ldiapp-5.  */

  PREPARE_GPR (0);
  /* Before stilp-0.  */
  __asm__ volatile ("stilp x19, x20, [x21]\n" : : : "memory");
  /* After stilp-0.  */

  PREPARE_GPR (0);
  /* Before stilp-1.  */
  __asm__ volatile ("stilp w19, w20, [x21]\n" : : : "memory");
  /* After stilp-1.  */

  PREPARE_GPR (16);
  /* Before stilp-2.  */
  __asm__ volatile ("stilp x19, x20, [x21, #-16]!\n" : : : "x21", "memory");
  /* After stilp-2.  */

  PREPARE_GPR (8);
  /* Before stilp-3.  */
  __asm__ volatile ("stilp w19, w20, [x21, #-8]!\n" : : : "x21", "memory");
  /* After stilp-3.  */
  /* Register overlap.  Since there is no offset, writeback is disabled.  */

  PREPARE_GPR (0);
  /* Before stilp-4.  */
  __asm__ volatile ("stilp x21, x20, [x21]\n" : : : "memory");
  /* After stilp-4.  */

  PREPARE_GPR (0);
  /* Before stilp-5.  */
  __asm__ volatile ("stilp w21, w20, [x21]\n" : : : "memory");
  /* After stilp-5.  */

  PREPARE_GPR (0);
  /* Before ldapr-0.  */
  __asm__ volatile ("ldapr x19, [x21], #8\n" : : : "x19", "x21", "memory");
  /* After ldapr-0.  */

  PREPARE_GPR (0);
  /* Before ldapr-1.  */
  __asm__ volatile ("ldapr w19, [x21], #4\n" : : : "x19", "x21", "memory");
  /* After ldapr-1.  */

  PREPARE_GPR (8);
  /* Before stlr-0.  */
  __asm__ volatile ("stlr x19, [x21, #-8]!\n" : : : "x21", "memory");
  /* After stlr-0.  */

  PREPARE_GPR (4);
  /* Before stlr-1.  */
  __asm__ volatile ("stlr w19, [x21, #-4]!\n" : : : "x21", "memory");
  /* After stlr-1.  */

  PREPARE_VECTOR_REG (0);
  /* Before ldap1-0.  */
  __asm__ volatile ("ldap1 {v22.d}[0], [x21]\n" : : : "v22", "memory");
  /* After ldap1-0.  */

  PREPARE_VECTOR_REG (0);
  /* Before stl1-0.  */
  __asm__ volatile ("stl1 {v22.d}[0], [x21]\n" : : : "memory");
  /* After stl1-0.  */

  PREPARE_VECTOR_REG (0);
  /* Before ldapur-0.  */
  __asm__ volatile ("ldapur d22, [x21]\n" : : : "v22", "memory");
  /* After ldapur-0.  */

  PREPARE_VECTOR_REG (0);
  /* Before stlur-0.  */
  __asm__ volatile ("stlur d22, [x21]\n" : : : "memory");
  /* After stlur-0.  */

  PREPARE_VECTOR_REG (256);
  /* Before ldapur-1.  */
  __asm__ volatile ("ldapur d22, [x21, #-256]\n" : : : "v22", "memory");
  /* After ldapur-1.  */

  PREPARE_VECTOR_REG (256);
  /* Before stlur-1.  */
  __asm__ volatile ("stlur d22, [x21, #-256]\n" : : : "memory");
  /* After stlur-1.  */

  PREPARE_VECTOR_REG (-255);
  /* Before ldapur-2.  */
  __asm__ volatile ("ldapur d22, [x21, #255]\n" : : : "v22", "memory");
  /* After ldapur-2.  */

  PREPARE_VECTOR_REG (-255);
  /* Before stlur-2.  */
  __asm__ volatile ("stlur d22, [x21, #255]\n" : : : "memory");
  /* After stlur-2.  */

  PREPARE_VECTOR_REG (0);
  /* Before ldapur-3.  */
  __asm__ volatile ("ldapur h22, [x21]\n" : : : "v22", "memory");
  /* After ldapur-3.  */

  PREPARE_VECTOR_REG (0);
  /* Before stlur-3.  */
  __asm__ volatile ("stlur h22, [x21]\n" : : : "memory");
  /* After stlur-3.  */

  PREPARE_VECTOR_REG (0);
  /* Before ldapur-4.  */
  __asm__ volatile ("ldapur s22, [x21]\n" : : : "v22", "memory");
  /* After ldapur-4.  */

  PREPARE_VECTOR_REG (0);
  /* Before stlur-4.  */
  __asm__ volatile ("stlur s22, [x21]\n" : : : "memory");
  /* After stlur-4.  */

  PREPARE_VECTOR_REG (0);
  /* Before ldapur-5.  */
  __asm__ volatile ("ldapur d22, [x21]\n" : : : "v22", "memory");
  /* After ldapur-5.  */

  PREPARE_VECTOR_REG (0);
  /* Before stlur-5.  */
  __asm__ volatile ("stlur d22, [x21]\n" : : : "memory");
  /* After stlur-5.  */

  PREPARE_VECTOR_REG (0);
  /* Before ldapur-6.  */
  __asm__ volatile ("ldapur q22, [x21]\n" : : : "v22", "memory");
  /* After ldapur-6.  */

  PREPARE_VECTOR_REG (0);
  /* Before stlur-6.  */
  __asm__ volatile ("stlur q22, [x21]\n" : : : "memory");
  /* After stlur-6.  */

  return 0;
}
