	# Two adjacent instruction ranges.  R_XTENSA_32_ABS keeps the address in
	# r_addend; the relocated words are non-zero and must not be added.
	# Adding them makes the ranges non-adjacent, so the linker will not
	# merge the entries.
	.section .xt.prop
	.reloc ., R_XTENSA_32_ABS, .text
	.word 0x10
	.word 4
	.word 2			# XTENSA_PROP_INSN

	.reloc ., R_XTENSA_32_ABS, .text + 4
	.word 0x20
	.word 4
	.word 2			# XTENSA_PROP_INSN

	.text
	.global _start
_start:
	.skip 8
