	.section .text.zz_support,"ax",@progbits
	.globl	near_callee
near_callee:
	rtsd	r15, 8
	nop
	.globl	_start
_start:
	brlid	r15, aaa_relaxed
	nop
	brlid	r15, zzz_victim
	nop
	bri	0
	.data
	.globl	gvar
	.align	2
gvar:	.space	64
