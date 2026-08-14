	.text
	.seh_proc	foo
foo:
	.seh_stackalloc 16
	.seh_endprologue
	.rept 2100000 / 4
	nop
	.endr
	.seh_handler _ZN9exception6handleEPvS0_S0_S0_, @except
	.seh_handlerdata
	.long 1
	.seh_code
	.seh_endproc
