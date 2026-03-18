# Source file used to test illegal opcodes.

foo:
	add	r1, r2, 101
	jmp	r2
	xin	0, r10, 1
# Not available in V1
	sxin	0, r10, 1
