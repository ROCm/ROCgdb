	.text
foo:
	movq	foo@GOTPCREL(%rip), %rax
	push	foo@GOTPCREL + eight(%rip)
	push	foo@GOTPCREL - eight(%rip)

	.equ eight, 8
