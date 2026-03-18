# Source file used to test illegal opcodes.

foo:
	add	r1, r2, 101
	jmp	r2
# Not available in V1
	xin	0, r10, 1
