# Check 64bit AVX10_VPMLDSABFLYDQ instructions

	.text
_start:
	vpmldsabflydq	$123, %xmm28, %xmm29, %xmm30
	vpmldsabflydq	$123, %ymm28, %ymm29, %ymm30
	vpmldsabflydq	$123, %zmm28, %zmm29, %zmm30
	vpmldsabflydq	$123, 0x10000000(%rbp, %r14, 8), %xmm29, %xmm30
	vpmldsabflydq	$123, (%r9), %xmm29, %xmm30
	vpmldsabflydq	$123, 2032(%rcx), %xmm29, %xmm30
	vpmldsabflydq	$123, -2048(%rdx), %xmm29, %xmm30
	vpmldsabflydq	$123, 0x10000000(%rbp, %r14, 8), %ymm29, %ymm30
	vpmldsabflydq	$123, (%r9), %ymm29, %ymm30
	vpmldsabflydq	$123, 4064(%rcx), %ymm29, %ymm30
	vpmldsabflydq	$123, -4096(%rdx), %ymm29, %ymm30
	vpmldsabflydq	$123, 0x10000000(%rbp, %r14, 8), %zmm29, %zmm30
	vpmldsabflydq	$123, (%r9), %zmm29, %zmm30
	vpmldsabflydq	$123, 8128(%rcx), %zmm29, %zmm30
	vpmldsabflydq	$123, -8192(%rdx), %zmm29, %zmm30

_intel:
	.intel_syntax noprefix
	vpmldsabflydq	xmm30, xmm29, xmm28, 123
	vpmldsabflydq	ymm30, ymm29, ymm28, 123
	vpmldsabflydq	zmm30, zmm29, zmm28, 123
	vpmldsabflydq	xmm30, xmm29, XMMWORD PTR [rbp+r14*8+0x10000000], 123
	vpmldsabflydq	xmm30, xmm29, XMMWORD PTR [r9], 123
	vpmldsabflydq	xmm30, xmm29, XMMWORD PTR [rcx+2032], 123
	vpmldsabflydq	xmm30, xmm29, XMMWORD PTR [rdx-2048], 123
	vpmldsabflydq	ymm30, ymm29, YMMWORD PTR [rbp+r14*8+0x10000000], 123
	vpmldsabflydq	ymm30, ymm29, YMMWORD PTR [r9], 123
	vpmldsabflydq	ymm30, ymm29, YMMWORD PTR [rcx+4064], 123
	vpmldsabflydq	ymm30, ymm29, YMMWORD PTR [rdx-4096], 123
	vpmldsabflydq	zmm30, zmm29, ZMMWORD PTR [rbp+r14*8+0x10000000], 123
	vpmldsabflydq	zmm30, zmm29, ZMMWORD PTR [r9], 123
	vpmldsabflydq	zmm30, zmm29, ZMMWORD PTR [rcx+8128], 123
	vpmldsabflydq	zmm30, zmm29, ZMMWORD PTR [rdx-8192], 123
