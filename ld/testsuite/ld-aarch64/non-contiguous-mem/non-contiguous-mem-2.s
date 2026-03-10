	.section .bss.MY_BUF, "aw", %nobits
	.global MY_BUF
	.type MY_BUF, %object
MY_BUF:
	.space 102400 /* 100KB */
	.size MY_BUF, . - MY_BUF


	.section .text.foo,"ax",%progbits
	.global foo
	.type	foo, %function
foo:
	ldr x0, .L3
	ret
.L3:
	.word	MY_BUF
	.size	foo, .-foo
