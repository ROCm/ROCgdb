	.section .text

	.globl _start
	.type _start, %function
_start:
	.rept 50
	call target
	call target + 4
	call target + 8
	.endr

	.globl target
	.type target, %function
target:
	.word 0x00000013
	.globl target_4
target_4:
	.word 0x00000013
	.globl target_8
target_8:
	ret
