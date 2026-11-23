	.text
	.type	resolver, %function
resolver:
	ret
	.globl	ifn
	.type	ifn, %gnu_indirect_function
	.set	ifn, resolver
	.globl	_start
_start:
	call	ifn
	ret
	.section .debug_info,"",%progbits
	.dword	ifn
