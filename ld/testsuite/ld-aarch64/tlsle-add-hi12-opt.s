	.global	test
	.section	.tbss,"awT",%nobits
tlsvar:
	.zero	4080
tlsvar_hi12:
	.zero	8

	.text
test:
	add	x0, x1, #:tprel_hi12:tlsvar
	add	x2, sp, #:tprel_hi12:tlsvar
	add	sp, x3, #:tprel_hi12:tlsvar
	add	x4, x5, #:tprel_hi12:tlsvar_hi12
