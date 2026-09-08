	.text
# Invalid 16-bit broadcast with EVEX.W == 1.
	.byte 0x62, 0xc3, 0x8c, 0x1d, 0x66, 0x90, 0x66, 0x90, 0x66, 0x90
# Invalid vcvtsi2sd with EVEX.b == 1.
	.byte 0x62,0xc1,0xff,0x38,0x2a,0x20
# Broadcast is invalid for stores.
	.insn EVEX.f3.0f.W1 0x7f, %zmm1, (%ecx){1to8}
# Invalid vmovdqu{8,16,64} with broadcast.
	.insn EVEX.f2.0f.W0 0x6f, (%ecx){1to16}, %zmm1
	.insn EVEX.f2.0f.W1 0x6f, (%ecx){1to8}, %zmm1
	.insn EVEX.f3.0f.W1 0x6f, (%ecx){1to8}, %zmm1
