	.section .rodata.probe.str1.1,"aMS",@progbits,1
.LCv:
	.string "v"

	.section .text.other,"ax",@progbits
	.literal_position
	.literal .Lb, .LCv
	.align 4
	.global other
other:
	entry	a1, 32
	l32r	a2, .Lb
	retw
