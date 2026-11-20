	.section .text

	.globl _start
	.type _start, %function
_start:
	.rept 5
	tail target
	.endr
	ret
	.space 4096

	.globl __jvt_base$
__jvt_base$:
	.word 0

	.type target, %function
target:
	ret
