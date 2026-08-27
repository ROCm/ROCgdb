	.globl	_start
	.globl	foo
	.text
	.align	4
_start:
	.literal	.Lunused, 0xffffffff
	entry	a5, 16
.L1:
	.space	60
	.begin	longcalls
	.rept	16
	call4	foo
	.endr
	.end	longcalls
.L2:

	.section	.debug_info, "", @progbits
	.uleb128	.L2 - .L1
