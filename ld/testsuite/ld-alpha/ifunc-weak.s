	.text

	# A weak IFUNC, both called and referenced.  It is not preemptible
	# in an executable, so this link resolves it itself, but its weak
	# binding makes check_relocs guess that it might become dynamic.
	.weak	weak_ifunc
	.type	weak_ifunc, @gnu_indirect_function
weak_ifunc:
	ret

	.globl	_start
	.ent	_start
_start:
	ldgp	$29, 0($27)
	ldq	$27, weak_ifunc($29)	!literal!1
	jsr	$26, ($27), 0		!lituse_jsr!1
	ldgp	$29, 0($26)
	ret
	.end	_start

	.data
	.globl	ptr
ptr:
	.quad	weak_ifunc
