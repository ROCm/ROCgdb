	# GNU as emits R_XTENSA_32, which is partial_inplace, so the addend
	# may be held in the relocated word rather than in r_addend.
	.section .rodata
legacy:
	.word 8
	.reloc legacy, R_XTENSA_32, target
