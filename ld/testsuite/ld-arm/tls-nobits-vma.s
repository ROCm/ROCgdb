	.section .data,"aw",%progbits
	.word 1

	.section .tdata,"awT",%progbits
	.word 2

	.section .tbss,"awT",%nobits
	.space 4

	.section .bss,"aw",%nobits
	.space 8
