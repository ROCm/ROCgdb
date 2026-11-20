	.section .text.live, "ax"

	.globl _start
	.type _start, %function
_start:
	ret

	.section .text.dead_group, "axG", @progbits, dead_group, comdat
	.type dead_group_start, %function
dead_group_start:
	.rept 70
	call dead_group_target
	.endr
	ret

	.type dead_group_target, %function
dead_group_target:
	ret
