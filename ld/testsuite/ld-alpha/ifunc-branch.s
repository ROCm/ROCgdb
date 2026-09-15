	.text

	.globl	global_ifunc
	.type	global_ifunc, @gnu_indirect_function
global_ifunc:
	ret

	# A direct branch reaches the resolver rather than the function it
	# selects, so it cannot be used to call an IFUNC.
	.globl	_start
	.ent	_start
_start:
	ldgp	$29, 0($27)
	bsr	$26, global_ifunc
	bsr	$26, global_ifunc	!samegp
	ret
	.end	_start
