# Linker relaxation must not disturb the addend of an R_MICROBLAZE_64
# relocation that refers to a symbol outside the section being relaxed.
#
# .text.aaa_relaxed contains an IMM that relaxation deletes.  .text.zzz_victim,
# a sibling section in the same object file, contains
#
#	R_MICROBLAZE_64  gvar + 0x18
#
# While relaxing .text.aaa_relaxed, microblaze_elf_relax_section() walks the
# relocations of every other section of the same BFD and, for R_MICROBLAZE_64,
# indexes isymbuf with ELF32_R_SYM (irelscan->r_info) without checking it
# against symtab_hdr->sh_info.  gvar is global, so its symbol index is >=
# sh_info and the read is out of bounds -- isymbuf holds only sh_info entries
# once --gc-sections has installed the locals-only cache.  If the bytes read
# happen to satisfy
#
#	isym->st_shndx == shndx && ELF32_ST_TYPE (isym->st_info) == STT_SECTION
#
# the addend is decremented by calc_fixup(), and gvar + 0x18 links as
# gvar + 0x14.
#
# The out-of-bounds read itself is unconditional; run ld under ASan or
# valgrind to observe it.  Whether it corrupts the addend depends on what is
# in the adjacent heap, so the check below is a correctness assertion rather
# than a reliable trigger.

	.section .text.aaa_relaxed,"ax",@progbits
	.globl	aaa_relaxed
	.type	aaa_relaxed,@function
aaa_relaxed:
	addik	r1, r1, -28
	swi	r15, r1, 0
	brlid	r15, near_callee	# IMM + BRLID; the IMM is deleted
	nop
	lwi	r15, r1, 0
	rtsd	r15, 8
	addik	r1, r1, 28
	.size	aaa_relaxed, .-aaa_relaxed

	.section .text.zzz_victim,"ax",@progbits
	.globl	zzz_victim
	.type	zzz_victim,@function
zzz_victim:
	lwi	r3, r0, gvar+24		# IMM + LWI; R_MICROBLAZE_64 gvar+0x18
	rtsd	r15, 8
	nop
	.size	zzz_victim, .-zzz_victim

/* A data reference to the same symbol with the same non-zero addend.  This
   goes through the R_MICROBLAZE_32 arm of the same loop, and unlike the
   instruction above it can be checked with readelf alone.  */
	.section .checkdata,"aw",@progbits
	.globl	check_word
check_word:
	.4byte	gvar + 0x18
