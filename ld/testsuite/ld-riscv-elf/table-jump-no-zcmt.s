	.section .text

	.globl _start
	.type	_start, %function
_start:
	# This file is assembled without Zcmt extension.
	# The linker should NOT generate JVT section
	.rept 50
	call foo
	.endr
	ret

	.globl foo
	.type	foo, %function
foo:
	ret
