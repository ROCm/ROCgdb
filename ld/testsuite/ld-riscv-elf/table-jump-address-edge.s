	.text
	.globl _start
_start:
	lui a0,%hi(x0_target)
	addi a0,a0,%lo(x0_target)
	lui a1,%hi(gp_target)
	addi a1,a1,%lo(gp_target)
	.rept 5
	tail far_target
	.endr

	.section .far,"ax",@progbits
far_target:
	ret

	.section .data.x0,"aw",@progbits
x0_target:
	.byte 1
	.section .data.gp,"aw",@progbits
gp_target:
	.byte 2
