	.text

	.globl	global_ifunc
	.type	global_ifunc, @gnu_indirect_function
global_ifunc:
	ret

	.globl	_start
	.ent	_start
_start:
	ret
	.end	_start

	# A gp-relative offset to an IFUNC is an offset to the resolver
	# rather than to the function it selects.
	.data
	.gprel32 global_ifunc
