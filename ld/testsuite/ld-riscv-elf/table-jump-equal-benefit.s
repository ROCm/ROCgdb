	.section .text

	.globl _start
	.type _start, %function
_start:
	.rept 70
	call func_b
	call func_a
	.endr

	.globl func_a
	.type func_a, %function
func_a:
	ret

	.globl func_b
	.type func_b, %function
func_b:
	ret
