	.globl	_start
	.globl	foo
	.text
	.align	4
_start:
	.literal	.Lunused, 0xffffffff
	entry	a5, 16
.L1:
	.begin	longcalls
	call4	foo
	.end	longcalls
	nop
.L2:

	.section	.debug_info, "", @progbits
	.uleb128	.L2 - .L1
