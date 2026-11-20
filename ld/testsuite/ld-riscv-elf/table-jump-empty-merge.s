	.text
	.globl _start
_start:	tail done
done:	ret

	.section .strings.1,"aMS",@progbits,1
	.asciz "abc", "abc", "xyz"
	.section .strings.2,"aMS",@progbits,1
	.asciz "xyz", "abc"
	.ifdef MIXED
	.section .extra,"a",@progbits
	.byte 0x21
	.endif
