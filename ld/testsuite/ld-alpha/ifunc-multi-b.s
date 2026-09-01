	.text

	.globl	other
	.ent	other
other:
	ldgp	$29, 0($27)
	ldq	$27, global_ifunc($29)	!literal!1
	jsr	$26, ($27), 0		!lituse_jsr!1
	ldgp	$29, 0($26)
	ret
	.end	other

	.data
	.globl	ptr_b
ptr_b:
	.quad	global_ifunc
