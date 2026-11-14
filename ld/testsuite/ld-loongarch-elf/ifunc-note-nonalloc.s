	.text
	.type	resolver, %function
resolver:
	jr	$ra
	.type	ifn, %gnu_indirect_function
	.set	ifn, resolver
	.globl	_start
_start:
	jr	$ra
	.section .note.probe,"",%note
	.dword	ifn
