	.text

	# One object defines the IFUNC and both call and reference it, so
	# that the GOT entries merge into one and the two data references
	# accumulate on the same hash table entry.
	.globl	global_ifunc
	.type	global_ifunc, @gnu_indirect_function
global_ifunc:
	ret

	.globl	_start
	.ent	_start
_start:
	ldgp	$29, 0($27)
	ldq	$27, global_ifunc($29)	!literal!1
	jsr	$26, ($27), 0		!lituse_jsr!1
	ldgp	$29, 0($26)
	ret
	.end	_start

	.data
	.globl	ptr_a
ptr_a:
	.quad	global_ifunc
