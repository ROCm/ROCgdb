# Check 64-bit ACE V1 instructions

	.text
start:
	tilezero %tmm0
	ldtilecfg (%rcx)
	sttilecfg (%rcx)
	tilerelease

	tilemovrow %ebx, %tmm2, %zmm1
	tilemovrow $8, %tmm2, %zmm1
	tilemovrow %ebx, %zmm2, %tmm1
	tilemovrow $8, %zmm2, %tmm1
	tilemovcol %ebx, %zmm2, %tmm1
	tilemovcol $8, %zmm2, %tmm1

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

	top4mxbf8ps  $0, %zmm2, %zmm1, %tmm1
	top4mxbhf8ps $0, %zmm2, %zmm1, %tmm1
	top4mxhbf8ps $0, %zmm2, %zmm1, %tmm1
	top4mxhf8ps  $0, %zmm2, %zmm1, %tmm1

	top4mxbssps $0, %zmm2, %zmm1, %tmm1

	top2bf16ps %zmm2, %zmm1, %tmm0

	top4bssd   %zmm2, %zmm1, %tmm0
	top4bsud   %zmm2, %zmm1, %tmm0
	top4busd   %zmm2, %zmm1, %tmm0
	top4buud   %zmm2, %zmm1, %tmm0
