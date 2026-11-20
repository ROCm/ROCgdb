# Check 32bit AVX10_VPMLDSABFLYDQ instructions

	.text
_start:
	vpmldsabflydq	$123, %xmm4, %xmm5, %xmm6
	vpmldsabflydq	$123, %ymm4, %ymm5, %ymm6
	vpmldsabflydq	$123, %zmm4, %zmm5, %zmm6
	vpmldsabflydq	$123, 0x10000000(%esp, %esi, 8), %xmm5, %xmm6
	vpmldsabflydq	$123, (%ecx), %xmm5, %xmm6
	vpmldsabflydq	$123, 2032(%ecx), %xmm5, %xmm6
	vpmldsabflydq	$123, -2048(%edx), %xmm5, %xmm6
	vpmldsabflydq	$123, 0x10000000(%esp, %esi, 8), %ymm5, %ymm6
	vpmldsabflydq	$123, (%ecx), %ymm5, %ymm6
	vpmldsabflydq	$123, 4064(%ecx), %ymm5, %ymm6
	vpmldsabflydq	$123, -4096(%edx), %ymm5, %ymm6
	vpmldsabflydq	$123, 0x10000000(%esp, %esi, 8), %zmm5, %zmm6
	vpmldsabflydq	$123, (%ecx), %zmm5, %zmm6
	vpmldsabflydq	$123, 8128(%ecx), %zmm5, %zmm6
	vpmldsabflydq	$123, -8192(%edx), %zmm5, %zmm6

_intel:
	.intel_syntax noprefix
	vpmldsabflydq	xmm6, xmm5, xmm4, 123
	vpmldsabflydq	ymm6, ymm5, ymm4, 123
	vpmldsabflydq	zmm6, zmm5, zmm4, 123
	vpmldsabflydq	xmm6, xmm5, XMMWORD PTR [esp+esi*8+0x10000000], 123
	vpmldsabflydq	xmm6, xmm5, [ecx], 123
	vpmldsabflydq	xmm6, xmm5, XMMWORD PTR [ecx+2032], 123
	vpmldsabflydq	xmm6, xmm5, [edx-2048], 123
	vpmldsabflydq	ymm6, ymm5, [esp+esi*8+0x10000000], 123
	vpmldsabflydq	ymm6, ymm5, YMMWORD PTR [ecx], 123
	vpmldsabflydq	ymm6, ymm5, [ecx+4064], 123
	vpmldsabflydq	ymm6, ymm5, YMMWORD PTR [edx-4096], 123
	vpmldsabflydq	zmm6, zmm5, ZMMWORD PTR [esp+esi*8+0x10000000], 123
	vpmldsabflydq	zmm6, zmm5, [ecx], 123
	vpmldsabflydq	zmm6, zmm5, [ecx+8128], 123
	vpmldsabflydq	zmm6, zmm5, ZMMWORD PTR [edx-8192], 123
