	.text
	.globl main
	.align 4
main:
	loopgtz a9, .Lloop_end
	.rep 200
	nop
	.endr
.Lloop_end:
	nop
