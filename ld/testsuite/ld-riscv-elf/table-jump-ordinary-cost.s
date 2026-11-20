	.section .text.start,"ax",@progbits
	.globl _start
_start:
	.rept 5
	tail short_target
	.endr
	.rept 3
	tail medium_target
	.endr
	.section .text.targets,"ax",@progbits
short_target:
	ret
	.space 5000
medium_target:
	ret
