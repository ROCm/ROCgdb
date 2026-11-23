	.text
	.type	resolver, %function
resolver:
	ret
	.type	ifn, %gnu_indirect_function
	.set	ifn, resolver
	.globl	_start
_start:
	ret
	.section .note.probe,"",%note
	.dword	ifn
