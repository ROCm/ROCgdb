	.text
	.globl	_start
_start:
	ret

	.data
	.globl	var
var:
	.quad	0

	# A dynamic relocation against a global symbol in a read-only section.
	.section .rodata,"a",@progbits
	.quad	var
