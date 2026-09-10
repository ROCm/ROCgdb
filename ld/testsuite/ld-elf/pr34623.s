	.section .fini,"ax",%progbits
	.globl _fini
	.type _fini, %function
_fini:
	.space 4
	.section .init,"ax",%progbits
	.globl _init
	.type _init, %function
_init:
	.space 4
	.section .note.GNU-stack,"",%progbits
