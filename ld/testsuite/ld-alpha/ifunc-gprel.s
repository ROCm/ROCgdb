	.text

	.globl	global_ifunc
	.type	global_ifunc, @gnu_indirect_function
global_ifunc:
	ret

	# A gp-relative address is the address of the resolver rather than
	# of the function it selects.
	.globl	_start
	.ent	_start
_start:
	ldgp	$29, 0($27)
	ldah	$1, global_ifunc($29)	!gprelhigh
	lda	$1, global_ifunc($1)	!gprellow
	ldq	$1, global_ifunc($29)	!gprel
	ret
	.end	_start
