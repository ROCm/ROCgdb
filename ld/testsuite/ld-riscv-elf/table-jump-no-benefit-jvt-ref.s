	.section .text

	.globl _start
	.type _start, %function
_start:
	.option push
	.option norelax
	la a0, __jvt_base$
	.option pop
	tail target
	tail target
	.space 4096

	.type target, %function
target:
	ret
