	.section .tbss,"awT",%nobits
	.global tls_desc
tls_desc:
	.zero 8

	.text
	.global test
	.type test, %function
test:
	add	x0, x1, #:tlsdesc_lo12:tls_desc
	add	x2, x2, #:tlsdesc_lo12:tls_desc
	add	x3, sp, #:tlsdesc_lo12:tls_desc
	add	sp, x4, #:tlsdesc_lo12:tls_desc
	ret
	.size test, .-test

