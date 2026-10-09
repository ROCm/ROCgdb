# Check 64-bit ACE V1 instructions

	.text
start:
	tilezero %tmm0
	ldtilecfg (%rcx)
	sttilecfg (%rcx)
	tilerelease

	tilemovrow %ebx, %tmm2, %zmm1
	tilemovrow $8, %tmm2, %zmm1

	tcvtrowd2ps     %edx, %tmm5, %zmm30
	tcvtrowd2ps     $8, %tmm5, %zmm30

	.irp p, h, l
	tcvtrowps2bf16\p %edx, %tmm5, %zmm30
	tcvtrowps2bf16\p $8, %tmm5, %zmm30

	tcvtrowps2ph\p   %edx, %tmm5, %zmm30
	tcvtrowps2ph\p   $8, %tmm5, %zmm30

	bsrmov\p %zmm10, %bsr0
	bsrmov\p 8128(%rcx), %bsr0

	bsrmov\p %bsr0, %zmm4
	bsrmov\p %bsr0, -8192(%rcx)
	.endr

	bsrinit %bsr0
	bsrmovf %zmm3, %zmm1, %bsr0
	bsrmovf 8128(%rcx), %zmm1, %bsr0
