	.section .text.start, "ax"

	.globl _start
	.type _start, %function
_start:
	.rept 5
	tail target
	.endr
	ret

	.section .text.target, "ax"
	.type target, %function
target:
	ret
