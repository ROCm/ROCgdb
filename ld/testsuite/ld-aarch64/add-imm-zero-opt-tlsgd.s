	.section .tbss,"awT",%nobits
	.global tls_gd
tls_gd:
	.zero 8

	.text
	.global test
	.type test, %function
test:
	add	x0, x1, #:tlsgd_lo12:tls_gd
	add	x2, x2, #:tlsgd_lo12:tls_gd
	add	x3, sp, #:tlsgd_lo12:tls_gd
	add	sp, x4, #:tlsgd_lo12:tls_gd
	ret
	.size test, .-test

