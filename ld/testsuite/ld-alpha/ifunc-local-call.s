	.text

	# An IFUNC with local binding, called but not otherwise referenced,
	# so that only its GOT entry needs an IRELATIVE.
	.type	local_ifunc, @gnu_indirect_function
local_ifunc:
	ret

	.globl	_start
	.ent	_start
_start:
	ldgp	$29, 0($27)
	ldq	$27, local_ifunc($29)	!literal!1
	jsr	$26, ($27), 0		!lituse_jsr!1
	ldgp	$29, 0($26)
	ret
	.end	_start
