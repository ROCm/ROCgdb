	.section .text

	.globl foo
	.type	foo, %function
foo:
	# Shared library: JVT relaxation should be disabled
	.rept 50
	call bar
	.endr
	ret

	.globl bar
	.type	bar, %function
bar:
	ret
