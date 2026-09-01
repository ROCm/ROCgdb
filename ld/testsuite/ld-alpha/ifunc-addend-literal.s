	.text

	.globl	global_ifunc
	.type	global_ifunc, @gnu_indirect_function
global_ifunc:
	ret
	ret

	# An IRELATIVE's addend is the address of the resolver, so a
	# reference to an IFUNC cannot carry an offset.
	.globl	_start
	.ent	_start
_start:
	ldgp	$29, 0($27)
	ldq	$27, global_ifunc+4($29)	!literal!1
	jsr	$26, ($27), 0			!lituse_jsr!1
	ldgp	$29, 0($26)
	ret
	.end	_start
