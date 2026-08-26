	.text

_start:
	.irp pfx, "", {vex}, {evex}

	.arch default; .arch .nofma
	\pfx	vfmadd132ps	%xmm1, %xmm2, %xmm3
	\pfx	vfnmadd132pd	%ymm1, %ymm2, %ymm3
	\pfx	vfmsub213ps	%xmm1, %xmm2, %xmm3
	\pfx	vfnmsub213pd	%ymm1, %ymm2, %ymm3
	\pfx	vfmaddsub231ps	%xmm1, %xmm2, %xmm3
	\pfx	vfmsubadd231pd	%ymm1, %ymm2, %ymm3

	.arch default; .arch .nof16c
	\pfx	vcvtph2ps	%xmm1, %xmm2
	\pfx	vcvtph2ps	%xmm1, %ymm2
	\pfx	vcvtps2ph	$0, %xmm1, %xmm2
	\pfx	vcvtps2ph	$0, %ymm1, %xmm2

	.arch default; .arch .noavx512_vnni
	\pfx	vpdpbusd	%xmm1, %xmm2, %xmm3
	\pfx	vpdpwssd	%ymm1, %ymm2, %ymm3

	.arch default; .arch .noavx512vl
	\pfx	vpdpbusd	%xmm1, %xmm2, %xmm3
	\pfx	vpdpwssd	%ymm1, %ymm2, %ymm3

	.arch default; .arch .noavx_vnni_int8
	\pfx	vpdpbssd	%xmm1, %xmm2, %xmm3
	\pfx	vpdpbsuds	%xmm1, %xmm2, %xmm3
	\pfx	vpdpbuud	%ymm1, %ymm2, %ymm3

	.arch default; .arch .noavx_vnni_int16
	\pfx	vpdpwsud	%xmm1, %xmm2, %xmm3
	\pfx	vpdpwusds	%ymm1, %ymm2, %ymm3
	\pfx	vpdpwuud	%ymm1, %ymm2, %ymm3

	.endr

	ret
