/* Copyright (C) 2026 Free Software Foundation, Inc.

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

/* Hand-written DWARF for a function whose whole body came from a single
   inlined call, so that the DW_TAG_inlined_subroutine covers exactly the
   same range as the DW_TAG_subprogram containing it.  Two entries in the
   function table then have the same address range, and
   lookup_address_in_function_table has to break the tie.  It must pick the
   inlined routine -- the innermost one, and the one later in the DIE
   stream -- and it must pick the same one every time.

   Historically the tie was broken by comparing the two funcinfo pointers,
   which made the answer depend on the addresses bfd_zalloc happened to
   return.

   The CU declares an address size of 4 on every target, as dw2-ranges.S in
   this directory has long done.  That byte is the DWARF address_size, not
   the target's pointer size: every address here is zero, so the width it
   claims need only agree with the DWARF itself, and fixing it removes the
   last thing that varied from target to target.

   The zeroes are written with .fill rather than .4byte, and DW_AT_high_pc
   uses DW_FORM_data1, an offset from DW_AT_low_pc, rather than a second
   address.  That is deliberate.  Several assemblers align a multi-byte
   datum to its own size, or refuse it when it is misaligned: alpha and
   ia64 quietly pad, sh reports "misaligned data".  In a hand-written debug
   section that padding would land in the middle of the DWARF and corrupt
   it.  .fill emits bytes without going through that path -- and zero bytes
   are the same in either byte order -- so the debug sections come out with
   the same layout and field widths on every target, with only the
   multi-byte fields differing by target endianness.  The test needs no
   per-target variant.  */

	.text
	.globl outer_fn
	.type outer_fn, %function
outer_fn:
.Lfunc_begin:
	.4byte	0
	.4byte	0
.Lfunc_end:
	.size	outer_fn, .-outer_fn

/* The body is 8 bytes, so every DW_AT_high_pc below fits in the data1
   form that the abbrevs use.  Anything added here must keep it under
   256.  */

	.section .debug_info
.Lcu_begin:
	.4byte	.Lcu_end - .Lcu_start		/* Length of Compilation Unit */
.Lcu_start:
	.2byte	4				/* DWARF Version */
	.4byte	.Labbrev_begin			/* Offset into abbrev section */
	.byte	4				/* Address size */

	/* CU die */
	.uleb128	1			/* Abbrev: DW_TAG_compile_unit */
	.ascii		"dw2-inline-tie.c\0"	/* DW_AT_name */
	.byte		1			/* DW_AT_language = DW_LANG_C89 */
	.4byte		.Lline_begin		/* DW_AT_stmt_list */
	.fill		4, 1, 0			/* DW_AT_low_pc = 0 */
	.byte		.Lfunc_end - .Lfunc_begin	/* DW_AT_high_pc: offset from low_pc */

	/* The containing function.  */
	.uleb128	2			/* Abbrev: DW_TAG_subprogram */
	.ascii		"outer_fn\0"		/* DW_AT_name */
	.fill		4, 1, 0			/* DW_AT_low_pc = 0 */
	.byte		.Lfunc_end - .Lfunc_begin	/* DW_AT_high_pc: offset from low_pc */

	/* Inlined into it, over exactly the same range.  */
	.uleb128	3			/* Abbrev: DW_TAG_inlined_subroutine */
	.ascii		"inlined_fn\0"		/* DW_AT_name */
	.fill		4, 1, 0			/* DW_AT_low_pc = 0 */
	.byte		.Lfunc_end - .Lfunc_begin	/* DW_AT_high_pc: offset from low_pc */

	.byte		0			/* End of children of outer_fn */
	.byte		0			/* End of children of CU */
.Lcu_end:

	.section .debug_abbrev
.Labbrev_begin:
	.uleb128	1			/* Abbrev code */
	.uleb128	0x11			/* DW_TAG_compile_unit */
	.byte		1			/* has_children */
	.uleb128	0x3			/* DW_AT_name */
	.uleb128	0x8			/* DW_FORM_string */
	.uleb128	0x13			/* DW_AT_language */
	.uleb128	0xb			/* DW_FORM_data1 */
	.uleb128	0x10			/* DW_AT_stmt_list */
	.uleb128	0x17			/* DW_FORM_sec_offset */
	.uleb128	0x11			/* DW_AT_low_pc */
	.uleb128	0x1			/* DW_FORM_addr */
	.uleb128	0x12			/* DW_AT_high_pc */
	.uleb128	0xb			/* DW_FORM_data1 (offset from low_pc) */
	.byte		0x0			/* End of attributes */
	.byte		0x0

	.uleb128	2			/* Abbrev code */
	.uleb128	0x2e			/* DW_TAG_subprogram */
	.byte		1			/* has_children */
	.uleb128	0x3			/* DW_AT_name */
	.uleb128	0x8			/* DW_FORM_string */
	.uleb128	0x11			/* DW_AT_low_pc */
	.uleb128	0x1			/* DW_FORM_addr */
	.uleb128	0x12			/* DW_AT_high_pc */
	.uleb128	0xb			/* DW_FORM_data1 (offset from low_pc) */
	.byte		0x0			/* End of attributes */
	.byte		0x0

	.uleb128	3			/* Abbrev code */
	.uleb128	0x1d			/* DW_TAG_inlined_subroutine */
	.byte		0			/* has_children */
	.uleb128	0x3			/* DW_AT_name */
	.uleb128	0x8			/* DW_FORM_string */
	.uleb128	0x11			/* DW_AT_low_pc */
	.uleb128	0x1			/* DW_FORM_addr */
	.uleb128	0x12			/* DW_AT_high_pc */
	.uleb128	0xb			/* DW_FORM_data1 (offset from low_pc) */
	.byte		0x0			/* End of attributes */
	.byte		0x0

	.byte		0x0			/* End of abbrevs */

/* The line program is here because of the path under test, not because the
   test looks at line numbers.  comp_unit_find_nearest_line decodes the
   unit's line table first and gives up if that fails, so without a line
   program that decodes, lookup_address_in_function_table -- the tie-break
   this test is about -- is never reached at all.  */

	.section .debug_line
.Lline_begin:
	.4byte		.Lline_end - .Lline_start	/* Initial length */
.Lline_start:
	.2byte		2			/* Version */
	.4byte		.Lline_lines - .Lline_hdr	/* header_length */
.Lline_hdr:
	.byte		1			/* Minimum insn length */
	.byte		1			/* default_is_stmt */
	.byte		1			/* line_base */
	.byte		1			/* line_range */
	.byte		0x10			/* opcode_base */

	/* Standard opcode lengths */
	.byte		0
	.byte		1
	.byte		1
	.byte		1
	.byte		1
	.byte		0
	.byte		0
	.byte		0
	.byte		1
	.byte		0
	.byte		0
	.byte		1
	.byte		0
	.byte		0
	.byte		0

	/* Include directories */
	.byte		0

	/* File names */
	.ascii		"dw2-inline-tie.c\0"
	.uleb128	0
	.uleb128	0
	.uleb128	0

	.byte		0

.Lline_lines:
	.byte		0			/* Extended opcode */
	.uleb128	5			/* Length */
	.byte		2			/* DW_LNE_set_address */
	.fill		4, 1, 0			/* ... to 0 */

	.byte		3			/* DW_LNS_advance_line */
	.sleb128	16			/* ... to 17 */

	.byte		1			/* DW_LNS_copy */

	.byte		2			/* DW_LNS_advance_pc */
	.uleb128	.Lfunc_end - .Lfunc_begin

	.byte		0			/* Extended opcode */
	.uleb128	1			/* Length */
	.byte		1			/* DW_LNE_end_of_sequence */
.Lline_end:
