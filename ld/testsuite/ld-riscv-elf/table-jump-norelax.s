	.section .text

	.globl _start
	.type _start, %function
_start:
	.option push
	.option norelax
	call target
	tail target
	jal x0, target
	.option pop

	.space 4096

	.type target, %function
target:
	ret
