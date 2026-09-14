	.text

	.globl	_start
	.ent	_start
_start:
	ret
	.end	_start

	.data
	.globl	ptr
ptr:
	.quad	_start
