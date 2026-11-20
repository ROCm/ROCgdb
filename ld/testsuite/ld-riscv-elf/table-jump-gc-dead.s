	.section .text.live, "ax"

	.globl _start
	.type _start, %function
_start:
	ret

	.section .text.dead, "ax"
	.type dead_start, %function
dead_start:
	.rept 70
	call dead_target
	.endr
	ret

	.type dead_target, %function
dead_target:
	ret
