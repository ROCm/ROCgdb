# Check 64-bit AVX10 V2 AUX instructions

	.arch generic64
	.arch .avx10v2aux
	.text
_start:
	.irp m, bf8, bf8s, hf8, hf8s
	vcvtps2\m	%xmm1, %xmm0
	vcvtps2\m	%ymm1, %xmm0
	vcvtps2\m	%zmm1, %xmm0
	vcvtps2\m\()x	2032(%rcx), %xmm0
	vcvtps2\m\()y	4064(%rcx), %xmm0
	vcvtps2\m\()z	8128(%rcx), %xmm0
	vcvtps2\m	(%r9){1to4}, %xmm0
	vcvtps2\m	(%r9){1to8}, %xmm0
	vcvtps2\m	(%r9){1to16}, %xmm0
	vcvtps2\m	%xmm1, %xmm0{%k1}
	vcvtps2\m	%xmm1, %xmm0{%k1}{z}

	vcvtbiasps2\m	%xmm2, %xmm1, %xmm0
	vcvtbiasps2\m	%ymm2, %ymm1, %xmm0
	vcvtbiasps2\m	%zmm2, %zmm1, %xmm0
	vcvtbiasps2\m	2032(%rcx), %xmm1, %xmm0
	vcvtbiasps2\m	4064(%rcx), %ymm1, %xmm0
	vcvtbiasps2\m	8128(%rcx), %zmm1, %xmm0
	vcvtbiasps2\m	(%r9){1to4}, %xmm1, %xmm0
	vcvtbiasps2\m	(%r9){1to8}, %ymm1, %xmm0
	vcvtbiasps2\m	(%r9){1to16}, %zmm1, %xmm0
	vcvtbiasps2\m	%xmm2, %xmm1, %xmm0{%k1}
	vcvtbiasps2\m	%xmm2, %xmm1, %xmm0{%k1}{z}
	.endr

	.irp m, hf8, hf8s
	vcvtrops2\m	%xmm1, %xmm0
	vcvtrops2\m	%ymm1, %xmm0
	vcvtrops2\m	%zmm1, %xmm0
	vcvtrops2\m\()x	2032(%rcx), %xmm0
	vcvtrops2\m\()y	4064(%rcx), %xmm0
	vcvtrops2\m\()z	8128(%rcx), %xmm0
	vcvtrops2\m	(%r9){1to4}, %xmm0
	vcvtrops2\m	(%r9){1to8}, %xmm0
	vcvtrops2\m	(%r9){1to16}, %xmm0
	vcvtrops2\m	%xmm1, %xmm0{%k1}
	vcvtrops2\m	%xmm1, %xmm0{%k1}{z}
	.endr

	.irp f, bf, hf
	vcvt\f\()82ps	%xmm1, %xmm0
	vcvt\f\()82ps	%xmm1, %ymm0
	vcvt\f\()82ps	%xmm1, %zmm0
	vcvt\f\()82ps	508(%rcx), %xmm0
	vcvt\f\()82ps	1016(%rcx), %ymm0
	vcvt\f\()82ps	2032(%rcx), %zmm0
	vcvt\f\()82ps	%xmm1, %xmm0{%k1}
	vcvt\f\()82ps	%xmm1, %xmm0{%k1}{z}

	vcvt\f\()82bf4s	%xmm1, %xmm0
	vcvt\f\()82bf4s	%ymm1, %xmm0
	vcvt\f\()82bf4s	%zmm1, %ymm0
	vcvt\f\()82bf4s	%xmm1, 1016(%rcx)
	vcvt\f\()82bf4s	%ymm1, 2032(%rcx)
	vcvt\f\()82bf4s	%zmm1, 4064(%rcx)
	.endr

	vcvtbf82bf6s	%xmm1, %xmm0
	vcvtbf82bf6s	%ymm1, %ymm0
	vcvtbf82bf6s	%zmm1, %zmm0

	vcvthf82hf6s	%xmm1, %xmm0
	vcvthf82hf6s	%ymm1, %ymm0
	vcvthf82hf6s	%zmm1, %zmm0

	vcvtbf42hf8	%xmm1, %xmm0
	vcvtbf42hf8	%xmm1, %ymm0
	vcvtbf42hf8	%ymm1, %zmm0
	vcvtbf42hf8	1016(%rcx), %xmm0
	vcvtbf42hf8	2032(%rcx), %ymm0
	vcvtbf42hf8	4064(%rcx), %zmm0
	vcvtbf42hf8	%xmm1, %xmm0{%k1}
	vcvtbf42hf8	%xmm1, %xmm0{%k1}{z}

	.irp f, bf, hf
	vcvt\f\()62hf8	%xmm1, %xmm0
	vcvt\f\()62hf8	%ymm1, %ymm0
	vcvt\f\()62hf8	%zmm1, %zmm0
	vcvt\f\()62hf8	%xmm1, %xmm0{%k1}
	vcvt\f\()62hf8	%xmm1, %xmm0{%k1}{z}
	.endr

	vpmovssdb	%xmm1, %xmm0
	vpmovssdb	%ymm1, %xmm0
	vpmovssdb	%zmm1, %xmm0
	vpmovssdb	%xmm1, 508(%rcx)
	vpmovssdb	%ymm1, 1016(%rcx)
	vpmovssdb	%zmm1, 2032(%rcx)
	vpmovssdb	%xmm1, %xmm0{%k1}
	vpmovssdb	%xmm1, %xmm0{%k1}{z}

	vunpackb	$0x10, %xmm1, %xmm0
	vunpackb	$0x10, %ymm1, %ymm0
	vunpackb	$0x10, %zmm1, %zmm0
	vunpackb	$0x10, 2032(%rcx), %xmm0
	vunpackb	$0x10, 4064(%rcx), %ymm0
	vunpackb	$0x10, 8128(%rcx), %zmm0
	vunpackb	$0x10, %xmm1, %xmm0{%k1}
	vunpackb	$0x10, %xmm1, %xmm0{%k1}{z}

_intel:
	.intel_syntax noprefix

	.irp m, bf8, bf8s, hf8, hf8s
	vcvtps2\m	xmm0, xmm1
	vcvtps2\m	xmm0, ymm1
	vcvtps2\m	xmm0, zmm1
	vcvtps2\m	xmm0, XMMWORD PTR [rcx+2032]
	vcvtps2\m	xmm0, YMMWORD PTR [rcx+4064]
	vcvtps2\m	xmm0, ZMMWORD PTR [rcx+8128]
	vcvtps2\m	xmm0, DWORD PTR [r9]{1to4}
	vcvtps2\m	xmm0, DWORD PTR [r9]{1to8}
	vcvtps2\m	xmm0, DWORD PTR [r9]{1to16}
	vcvtps2\m	xmm0{k1}, xmm1
	vcvtps2\m	xmm0{k1}{z}, xmm1

	vcvtbiasps2\m	xmm0, xmm1, xmm2
	vcvtbiasps2\m	xmm0, ymm1, ymm2
	vcvtbiasps2\m	xmm0, zmm1, zmm2
	vcvtbiasps2\m	xmm0, xmm1, XMMWORD PTR [rcx+2032]
	vcvtbiasps2\m	xmm0, ymm1, YMMWORD PTR [rcx+4064]
	vcvtbiasps2\m	xmm0, zmm1, ZMMWORD PTR [rcx+8128]
	vcvtbiasps2\m	xmm0, xmm1, DWORD PTR [r9]{1to4}
	vcvtbiasps2\m	xmm0, ymm1, DWORD PTR [r9]{1to8}
	vcvtbiasps2\m	xmm0, zmm1, DWORD PTR [r9]{1to16}
	vcvtbiasps2\m	xmm0{k1}, xmm1, xmm2
	vcvtbiasps2\m	xmm0{k1}{z}, xmm1, xmm2
	.endr

	.irp m, hf8, hf8s
	vcvtrops2\m	xmm0, xmm1
	vcvtrops2\m	xmm0, ymm1
	vcvtrops2\m	xmm0, zmm1
	vcvtrops2\m	xmm0, XMMWORD PTR [rcx+2032]
	vcvtrops2\m	xmm0, YMMWORD PTR [rcx+4064]
	vcvtrops2\m	xmm0, ZMMWORD PTR [rcx+8128]
	vcvtrops2\m	xmm0, DWORD PTR [r9]{1to4}
	vcvtrops2\m	xmm0, DWORD PTR [r9]{1to8}
	vcvtrops2\m	xmm0, DWORD PTR [r9]{1to16}
	vcvtrops2\m	xmm0{k1}, xmm1
	vcvtrops2\m	xmm0{k1}{z}, xmm1
	.endr

	.irp f, bf, hf
	vcvt\f\()82ps	xmm0, xmm1
	vcvt\f\()82ps	ymm0, xmm1
	vcvt\f\()82ps	zmm0, xmm1
	vcvt\f\()82ps	xmm0, DWORD PTR [rcx+508]
	vcvt\f\()82ps	ymm0, QWORD PTR [rcx+1016]
	vcvt\f\()82ps	zmm0, XMMWORD PTR [rcx+2032]
	vcvt\f\()82ps	xmm0{k1}, xmm1
	vcvt\f\()82ps	xmm0{k1}{z}, xmm1

	vcvt\f\()82bf4s	xmm0, xmm1
	vcvt\f\()82bf4s	xmm0, ymm1
	vcvt\f\()82bf4s	ymm0, zmm1
	vcvt\f\()82bf4s	QWORD PTR [rcx+1016], xmm1
	vcvt\f\()82bf4s	XMMWORD PTR [rcx+2032], ymm1
	vcvt\f\()82bf4s	YMMWORD PTR [rcx+4064], zmm1
	.endr

	vcvtbf82bf6s	xmm0, xmm1
	vcvtbf82bf6s	ymm0, ymm1
	vcvtbf82bf6s	zmm0, zmm1

	vcvthf82hf6s	xmm0, xmm1
	vcvthf82hf6s	ymm0, ymm1
	vcvthf82hf6s	zmm0, zmm1

	vcvtbf42hf8	xmm0, xmm1
	vcvtbf42hf8	ymm0, xmm1
	vcvtbf42hf8	zmm0, ymm1
	vcvtbf42hf8	xmm0, QWORD PTR [rcx+1016]
	vcvtbf42hf8	ymm0, XMMWORD PTR [rcx+2032]
	vcvtbf42hf8	zmm0, YMMWORD PTR [rcx+4064]
	vcvtbf42hf8	xmm0{k1}, xmm1
	vcvtbf42hf8	xmm0{k1}{z}, xmm1

	.irp f, bf, hf
	vcvt\f\()62hf8	xmm0, xmm1
	vcvt\f\()62hf8	ymm0, ymm1
	vcvt\f\()62hf8	zmm0, zmm1
	vcvt\f\()62hf8	xmm0{k1}, xmm1
	vcvt\f\()62hf8	xmm0{k1}{z}, xmm1
	.endr

	vpmovssdb	xmm0, xmm1
	vpmovssdb	xmm0, ymm1
	vpmovssdb	xmm0, zmm1
	vpmovssdb	DWORD PTR [rcx+508], xmm1
	vpmovssdb	QWORD PTR [rcx+1016], ymm1
	vpmovssdb	XMMWORD PTR [rcx+2032], zmm1
	vpmovssdb	xmm0{k1}, xmm1
	vpmovssdb	xmm0{k1}{z}, xmm1

	vunpackb	xmm0, xmm1, 0x10
	vunpackb	ymm0, ymm1, 0x10
	vunpackb	zmm0, zmm1, 0x10
	vunpackb	xmm0, XMMWORD PTR [rcx+2032], 0x10
	vunpackb	ymm0, YMMWORD PTR [rcx+4064], 0x10
	vunpackb	zmm0, ZMMWORD PTR [rcx+8128], 0x10
	vunpackb	xmm0{k1}, xmm1, 0x10
	vunpackb	xmm0{k1}{z}, xmm1, 0x10
