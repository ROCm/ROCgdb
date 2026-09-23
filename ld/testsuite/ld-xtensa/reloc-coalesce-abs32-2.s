	# A literal as LLVM emits it: R_XTENSA_32_ABS with the addend in
	# r_addend and a zero literal word.
	.text
	.global foo
	.global g_name
	.align 4
	.literal .Lg_name, 0
	.reloc .Lg_name, R_XTENSA_32_ABS, g_name + 4
foo:
	entry a5,16
	movi a5,20000
	l32r a6,.Lg_name
	movi a7,50000
	ret
