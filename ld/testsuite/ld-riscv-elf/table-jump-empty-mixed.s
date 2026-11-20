	.text
	.globl _start
_start:	tail done
done:	ret

	.section .prefix,"a",@progbits
	.byte 0x11, 0x22, 0x33, 0x44
	.section .suffix,"a",@progbits
	.byte 0x55, 0x66, 0x77, 0x88
