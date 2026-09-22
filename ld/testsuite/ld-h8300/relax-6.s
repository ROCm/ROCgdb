	.h8300s
	.global _start, __start
_start:
__start:
	mov.b	r2l,@0xFFFFFFBD:32
	rts
