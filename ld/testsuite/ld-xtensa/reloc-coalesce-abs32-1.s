	.global foo
	.data
	.global g_name
	.align 4
g_name:
.Lg_name:
	.word 0xffffffff
	.text
	.global main
	.align 4
main:
	entry a5,16
	movi a5,20000
	# Through the section symbol, so R_XTENSA_32 holds the addend in the
	# literal word.
	movi a6,.Lg_name+4
	call8 foo
	ret
