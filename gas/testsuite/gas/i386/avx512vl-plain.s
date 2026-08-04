	.text
	.arch generic32
	.arch .avx512vl
_start:
	{evex} vaesenc		%ymm1, %ymm2, %ymm3
	       vcvtph2ps	%xmm1, %xmm2
	       vfmadd132ps	%xmm1, %xmm2, %xmm3
	       vgf2p8mulb	%ymm1, %ymm2, %ymm3{%k4}
	{evex} vpclmulqdq	$0, %ymm1, %ymm2, %ymm3

	.arch .vaes
	{evex} vaesenc		%ymm1, %ymm2, %ymm3

	.arch .gfni
	       vgf2p8mulb	%ymm1, %ymm2, %ymm3{%k4}

	.arch .vpclmulqdq
	{evex} vpclmulqdq	$0, %ymm1, %ymm2, %ymm3

	.arch generic32
	.arch .avx512f
	{evex}	vpermd		%ymm1, %ymm2, %ymm3
		vpermd		%ymm1, %ymm2, %ymm3
