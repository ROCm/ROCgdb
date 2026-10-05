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
	.endr
