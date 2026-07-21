	.section .tbss,"awT",%nobits
tls_ld:
	.zero 8

	.text
	.global test
	.type test, %function
test:
	add	x0, x1, #:tlsldm_lo12_nc:tls_ld
	add	x2, x2, #:tlsldm_lo12_nc:tls_ld
	add	x3, sp, #:tlsldm_lo12_nc:tls_ld
	add	sp, x4, #:tlsldm_lo12_nc:tls_ld
	ret
	.size test, .-test

