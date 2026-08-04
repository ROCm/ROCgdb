	.option arch, rv32i
	.option arch, +zve32x

eew32:
	# vle64.v v0, (x31)
	.insn 4, 0x020ff007

	.option arch, rv32i
	.option arch, +zve64x

eew64:
	vle64.v	v0, (x31)
