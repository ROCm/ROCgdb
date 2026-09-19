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

	.section .text
	.global main
main:
	nop

#ifdef __arm__
	.section	.text_1,"ax",%progbits
#else
	.section	.text_1,"ax",@progbits
#endif

	# This is foo, but we don't use a symbol.  We use a repetition of 8 to
	# get the end of the section to be 8-byte aligned, to increase the
	# chances of avoiding a section hole in between .text_1 and .text_2.
	.rept 8
	nop
	.endr

	.align 1

#ifdef __arm__
	.section	.text_2,"ax",%progbits
#else
	.section	.text_2,"ax",@progbits
#endif

	.global bar
bar:
	nop

#ifdef __arm__
	.section	.note.GNU-stack,"",%progbits
#else
	.section	.note.GNU-stack,"",@progbits
#endif
