	.section .text.grp,"axG",@progbits,grp,comdat
	.type	kept_ifunc, @gnu_indirect_function
kept_ifunc:
	ret

	.text
	.globl	_start
	.ent	_start
_start:
	ret
	.end	_start
