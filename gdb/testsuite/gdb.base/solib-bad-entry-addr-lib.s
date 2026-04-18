/* Copyright 2026 Free Software Foundation, Inc.

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

	.section .rodata
	.globl lib_var
	.type lib_var, @object
lib_var:
	.long 42
	.size lib_var, .-lib_var

	/* This executable section exists solely to force the linker
	   to create an additional LOAD segment (with R+X permissions),
	   giving the library 3+ LOAD segments instead of the default 2.

	   This matters because GDB's symfile_find_segment_sections only
	   runs for objects with exactly 1 or 2 segments.  When it runs,
	   it sets sect_index_text from the first section in segment 1,
	   which masks the bug we are testing for, see the .exp file for
	   details.  */
	.section .not_text, "ax", @progbits
	.byte 0
