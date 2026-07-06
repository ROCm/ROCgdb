	.section .bar,"aw",@progbits
	.p2align 3
	.dc.a	__ehdr_start@gotpcrel
	.dc.a	__ehdr_start

	.section .foo,"aw",@progbits
	.p2align 3
	.dc.a	__ehdr_start@gotpcrel
	.dc.a	__ehdr_start
	.section	.note.GNU-stack,"",@progbits
