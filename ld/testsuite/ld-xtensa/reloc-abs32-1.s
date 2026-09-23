	.text
	.global _start
	.global target
_start:
	.skip 2
target:
	.skip 2

	# Relocations as LLVM emits them: R_XTENSA_32_ABS with the addend in
	# r_addend and a zero relocated word.
	.section .rodata
	.reloc ., R_XTENSA_32_ABS, target
	.word 0
	.reloc ., R_XTENSA_32_ABS, target + 8
	.word 0
