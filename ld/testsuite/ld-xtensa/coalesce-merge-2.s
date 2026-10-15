	/* "{\"v\":7}" is 8 bytes.  .LC0+8 is one past the end and, before
	   merging, the same input offset as the following "v".  */
	.section .rodata.probe.str1.1,"aMS",@progbits,1
.LC0:
	.string "{\"v\":7}"
	.global gs
gs:
.LC1:
	.string "v"

	.section .text.probe,"ax",@progbits
	.literal_position
	.literal .Lbegin, .LC0
	.literal .Lend, .LC0+8
	.literal .Lkey, .LC1
	.literal .Lkey2, .LC1
	.literal .Lg, gs
	.align 4
	.global probe
probe:
	entry	a1, 32
	l32r	a2, .Lbegin
	l32r	a3, .Lend
	l32r	a4, .Lkey
	l32r	a5, .Lkey2
	l32r	a6, .Lg
	retw
