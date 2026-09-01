	.text

	.globl	_start
	.ent	_start
_start:
	ldgp	$29, 0($27)
	ldq	$27, shlib_ifunc($29)	!literal!1
	jsr	$26, ($27), 0		!lituse_jsr!1
	ldgp	$29, 0($26)
	ret
	.end	_start

	.data
	.globl	ptr
ptr:
	.quad	shlib_ifunc
