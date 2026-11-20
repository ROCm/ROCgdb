	.text
	.globl _start
_start:
	# Fill the JT slots using savings entirely before near_site.
	.irp n,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31
	.rept 4
	tail far\n
	.endr
	.endr
	.space 0x200000
	.irp n,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31
	.globl far\n
far\n:	ret
	.endr

	# With the default linker script, the JVT is between this site and
	# fini_fn, so it affects their displacement even though rewriting the
	# tail calls above moves both symbols towards lower addresses.
	.globl near_site
near_site:
	.ifdef TEST_CJAL
	call fini_fn
	.space 1980
	.else
	.ifdef TEST_BACKWARD
	ret
	.else
	tail fini_fn
	.endif
	.ifdef TEST_JAL
	.space 0x100000 - 128
	.else
	.space 1900
	.endif
	.endif
	.section .fini,"ax",@progbits
	.globl fini_fn
fini_fn:
	.ifdef TEST_BACKWARD
	tail near_site
	.else
	ret
	.endif
