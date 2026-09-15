	.text

	# A weak reference that carries the IFUNC type without a definition.
	# It resolves to zero like any other undefined weak function; the
	# type must not make the linker treat it as a defined IFUNC.
	.weak	weak_ifunc
	.type	weak_ifunc, @gnu_indirect_function

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
