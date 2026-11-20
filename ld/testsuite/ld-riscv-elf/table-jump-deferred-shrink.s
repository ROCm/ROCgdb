	.section .text.from,"ax",@progbits
	.globl _start
_start:
	# Shrinking the other transfers brings these calls into C.J range.
	# Counting their original widths would buy an unprofitable JT entry.
	.rept 8
	tail nearby
	.endr
	.space 2028
	.section .text.to,"ax",@progbits
nearby:
	ret
