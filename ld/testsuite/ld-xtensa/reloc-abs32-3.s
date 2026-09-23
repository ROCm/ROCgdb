	# A local label is referenced through the section symbol, so a
	# relocatable link must rebase the addend by the output offset of
	# this .text and keep the result in r_addend.
	.text
	.skip 4
.Llocal:
	.skip 4

	.section .rodata
	.reloc ., R_XTENSA_32_ABS, .Llocal + 4
	.word 0
