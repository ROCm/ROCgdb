	.section .tbss,"awT",%nobits
tls0:
	.zero 4096
tls_page:
	.zero 8

	.section .data.page,"aw"
	.p2align 12
page_sym:
	.xword 0

	.text
	.global test
	.type test, %function
test:
	add	x0, x0, #:lo12:page_sym
	add	x1, x2, #:dtprel_hi12:tls0, lsl #12
	add	x3, x3, #:dtprel_lo12:tls0
	add	x4, x5, #:dtprel_lo12_nc:tls_page
	add	x6, x7, #:tprel_hi12:tls0, lsl #12
	add	x8, x8, #:tprel_lo12:tls0-16
	add	x9, x10, #:tprel_lo12_nc:tls_page-16

	/* A 32-bit write must not be replaced with NOP.  */
	add	w11, w11, #:lo12:page_sym

	/* Uses of SP must remain ADD instructions.  */
	add	x12, sp, #:lo12:page_sym
	add	sp, x13, #:lo12:page_sym

	/* A nonzero immediate must remain an ADD instruction.  */
	add	x14, x14, #:lo12:page_sym+8
	ret
	.size test, .-test
