	.text

	# A globally visible IFUNC, both called and referenced, so that both
	# its GOT entry and the data word holding its address need an
	# IRELATIVE.
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
	.globl	ptr
ptr:
	.quad	global_ifunc
