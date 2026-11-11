	.text
	.globl	_start
_start:
	call	target
	.space	0xffffc

	.section	__lcxx_override,"ax",@progbits
	.globl	target
target:
	ret
