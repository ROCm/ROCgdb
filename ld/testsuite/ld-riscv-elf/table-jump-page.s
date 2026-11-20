	.section .from,"ax",@progbits
	.globl _start
_start:
	.ifdef TEST_JAL
	# x5 prevents the compressed or table-jump encoding on RV64.
	call t0,page_target
	.else
	tail page_target
	.endif

	.section .target,"ax",@progbits
page_target:
	ret

	.section .fillers,"ax",@progbits
	.rept 5
	tail far_target
	.endr
	.section .far,"ax",@progbits
far_target:
	ret
