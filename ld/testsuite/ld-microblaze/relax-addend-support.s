# Placed in .text.zz_support so that it is laid out after the sections of
# relax-addend.s and the call from aaa_relaxed is a forward reference.  A
# backward reference is not relaxed and the test would not exercise anything.

	.section .text.zz_support,"ax",@progbits
	.globl	near_callee
	.type	near_callee,@function
near_callee:
	rtsd	r15, 8
	nop
	.size	near_callee, .-near_callee

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
