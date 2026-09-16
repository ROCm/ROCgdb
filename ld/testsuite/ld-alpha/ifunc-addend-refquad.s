	.text

	.globl	global_ifunc
	.type	global_ifunc, @gnu_indirect_function
global_ifunc:
	ret
	ret

	.globl	_start
	.ent	_start
_start:
	ret
	.end	_start

	# An IRELATIVE's addend is the address of the resolver, so a
	# reference to an IFUNC cannot carry an offset.
	.data
	.globl	ptr
ptr:
	.quad	global_ifunc+4
