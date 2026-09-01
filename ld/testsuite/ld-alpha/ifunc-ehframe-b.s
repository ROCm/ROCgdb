	.text

	.globl	other
	.ent	other
other:
	.cfi_startproc
	.cfi_personality 0x00, global_ifunc
	ret
	.cfi_endproc
	.end	other
