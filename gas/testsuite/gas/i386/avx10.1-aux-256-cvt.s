# Check 32bit AVX10.1-aux/256 convert instructions

	.arch generic32
	.arch .avx10.1aux/256
	.equ AVX10_V1_AUX, 1
	.include "avx10_2-256-cvt.s"
