	.section .text

	.globl _start
	.type _start, %function
_start:
	tail target
	tail target
	.space 4096

	.type target, %function
target:
	ret
