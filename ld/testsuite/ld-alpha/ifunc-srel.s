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

	# A pc-relative offset to an IFUNC is an offset to the resolver
	# rather than to the function it selects.
	.data
	.short	global_ifunc - .
	.long	global_ifunc - .
	.quad	global_ifunc - .
