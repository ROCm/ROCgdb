	.section .text

	.globl _start
	.type	_start, %function
_start:
	# Test benefit ranking: higher frequency functions should
	# be prioritized for JVT entry allocation.
	# Use reverse alphabetical order to verify sorting by benefit,
	# not by name.

	# func_d called 80 times - highest benefit
	.rept 80
	call func_d
	.endr

	# func_c called 50 times - medium benefit
	.rept 50
	call func_c
	.endr

	# func_b called 20 times - lower benefit
	.rept 20
	call func_b
	.endr

	# func_a called 5 times - lowest benefit
	.rept 5
	call func_a
	.endr

	ret

	.globl func_a
	.type	func_a, %function
func_a:
	ret

	.globl func_b
	.type	func_b, %function
func_b:
	ret

	.globl func_c
	.type	func_c, %function
func_c:
	ret

	.globl func_d
	.type	func_d, %function
func_d:
	ret
