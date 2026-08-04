# Check 32bit AVX10.1-aux/512 convert instructions

	.arch generic32
	.arch .avx10.1aux/512
	.equ AVX10_V1_AUX, 1
	.include "avx10_2-512-cvt.s"
