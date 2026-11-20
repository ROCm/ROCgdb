	.text
	.globl _start
_start:
	# Ordinary relaxation reduces these transfers to C.J before the Zcmt
	# recording pass.  They must not contribute to table profitability.
	.rept 16
	tail target
	.endr
target:
	ret
