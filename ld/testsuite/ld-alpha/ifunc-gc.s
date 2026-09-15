	.section .text.ifunc,"ax",@progbits
	.globl	global_ifunc
	.type	global_ifunc, @gnu_indirect_function
global_ifunc:
	ret

	# Both of these need an IRELATIVE, but --gc-sections drops the
	# section holding the call, so only one may be reserved.
	.section .text.dead,"ax",@progbits
	.globl	dead
	.ent	dead
dead:
	ldgp	$29, 0($27)
	ldq	$27, global_ifunc($29)	!literal!1
	jsr	$26, ($27), 0		!lituse_jsr!1
	ldgp	$29, 0($26)
	ret
	.end	dead

	.section .data.live,"aw",@progbits
	.globl	ptr
ptr:
	.quad	global_ifunc

	.section .text.start,"ax",@progbits
	.globl	_start
	.ent	_start
_start:
	ldgp	$29, 0($27)
	ldah	$1, ptr($29)		!gprelhigh
	ret
	.end	_start
