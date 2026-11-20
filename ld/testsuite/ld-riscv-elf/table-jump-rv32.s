	.section .text

	# Test RV32 ZCMT relaxation for cm.jt
	# jal x0 (no link) -> cm.jt (index 0-31)
	# JVT entry cost: 4 bytes (RV32), benefit = 2 per call
	# Threshold: 3*2=6 > 4 (just profitable)

	.globl _start
	.type	_start, %function
_start:
	.rept 3
	tail foo_local
	tail foo_global
	.endr

	.space 4096

	.globl  foo_global
	.type	foo_global, %function
foo_global:
	ret

	.type	foo_local, %function
foo_local:
	ret
