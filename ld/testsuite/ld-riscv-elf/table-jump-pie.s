	.section .text

	.globl _start
	.type _start, %function
_start:
	.rept 70
	call target
	.endr
	ret

	.type target, %function
target:
	ret
