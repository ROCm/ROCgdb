/* dead_fn is dropped by --gc-sections; its FDE is then removed from .eh_frame,
   which shifts ehlocal and makes adjust_eh_frame_local_symbols() cache a
   locals-only symbol buffer in symtab_hdr->contents. */
	.section .text.dead,"ax",@progbits
	.globl	dead_fn
	.type	dead_fn,@function
dead_fn:
	rtsd	r15, 8
	nop
	.size	dead_fn, .-dead_fn

	.section .text.aaa_relaxed,"ax",@progbits
	.globl	aaa_relaxed
	.type	aaa_relaxed,@function
aaa_relaxed:
	addik	r1, r1, -28
	swi	r15, r1, 0
	brlid	r15, near_callee
	nop
	lwi	r15, r1, 0
	rtsd	r15, 8
	addik	r1, r1, 28
	.size	aaa_relaxed, .-aaa_relaxed

	.section .text.zzz_victim,"ax",@progbits
	.globl	zzz_victim
	.type	zzz_victim,@function
zzz_victim:
	lwi	r3, r0, gvar+24
	rtsd	r15, 8
	nop
	.size	zzz_victim, .-zzz_victim

	.section .eh_frame,"a",@progbits
/* Hand-assembled so that no label-difference expressions are used.  MicroBlaze
   GAS emits an R_MICROBLAZE_NONE marker for every resolved label difference,
   which lands in .rela.eh_frame and trips the
   BFD_ASSERT (cookie->rel->r_offset == ent->offset + 8) in
   _bfd_elf_discard_section_eh_frame.  Literal lengths and CIE pointers keep
   .rela.eh_frame to just the two FDE initial-location relocations. */

	/* CIE at 0x00, total 20 bytes */
	.4byte	16			/* length */
	.4byte	0			/* CIE id */
	.byte	1			/* version */
	.asciz	"zR"			/* augmentation */
	.uleb128 1			/* code alignment factor */
	.sleb128 -4			/* data alignment factor */
	.byte	15			/* return address register */
	.uleb128 1			/* augmentation data length */
	.byte	0x00			/* FDE encoding: DW_EH_PE_absptr */
	.byte	0x0c, 0x01, 0x00	/* DW_CFA_def_cfa r1, 0 */

	/* FDE for dead_fn at 0x14, total 20 bytes.  dead_fn is dropped by
	   --gc-sections, so this FDE is removed and everything after it moves. */
	.4byte	16			/* length */
	.4byte	0x18			/* CIE pointer: this field's offset - 0 */
	.4byte	dead_fn			/* initial location  <- the only reloc */
	.4byte	8			/* address range */
	.uleb128 0			/* augmentation data length */
	.byte	0, 0, 0			/* padding to 20 bytes */

ehlocal:				/* local symbol inside .eh_frame; it is
					   this symbol moving that makes
					   adjust_eh_frame_local_symbols() cache
					   a locals-only symbol buffer */

	/* FDE for aaa_relaxed at 0x28, total 20 bytes */
	.4byte	16
	.4byte	0x2c
	.4byte	aaa_relaxed		/* <- the only other reloc */
	.4byte	32
	.uleb128 0
	.byte	0, 0, 0

	.4byte	0			/* terminator */
