	.section .text.startup,"ax"
	.weak __jvt_base$
	.globl _start
	.type _start, %function
_start:
	.option nopic
	la a0, __jvt_base$
	beqz a0, 1f
	csrw jvt, a0
1:
	tail entry

	.section .text,"ax"
	.globl entry
	.type entry, %function
entry:
	.rept SITES
	tail foo_local
	tail foo_global
	.endr
	.space 4096

	.globl foo_global
	.type foo_global, %function
foo_global:
	ret

	.type foo_local, %function
foo_local:
	ret
