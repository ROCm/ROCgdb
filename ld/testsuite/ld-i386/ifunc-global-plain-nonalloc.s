	.text
	.type	resolver, @function
resolver:
	ret
	.globl	ifn
	.type	ifn, @gnu_indirect_function
	.set	ifn, resolver
	.globl	_start
_start:
	ret
	.section .probe,"",@progbits
	.long	ifn
