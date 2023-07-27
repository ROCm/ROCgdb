# Check 32bit AVX-VNNI-INT16 instructions

	.text
_start:
	vpdpwsud	%ymm4, %ymm5, %ymm6	 #AVX-VNNI-INT16
	vpdpwsud	%xmm4, %xmm5, %xmm6	 #AVX-VNNI-INT16
	vpdpwsud	0x10000000(%esp, %esi, 8), %ymm5, %ymm6	 #AVX-VNNI-INT16
	vpdpwsud	(%ecx), %ymm5, %ymm6	 #AVX-VNNI-INT16
	vpdpwsud	4064(%ecx), %ymm5, %ymm6	 #AVX-VNNI-INT16 Disp32(e00f0000)
	vpdpwsud	-4096(%edx), %ymm5, %ymm6	 #AVX-VNNI-INT16 Disp32(00f0ffff)
	vpdpwsud	0x10000000(%esp, %esi, 8), %xmm5, %xmm6	 #AVX-VNNI-INT16
	vpdpwsud	(%ecx), %xmm5, %xmm6	 #AVX-VNNI-INT16
	vpdpwsud	2032(%ecx), %xmm5, %xmm6	 #AVX-VNNI-INT16 Disp32(f0070000)
	vpdpwsud	-2048(%edx), %xmm5, %xmm6	 #AVX-VNNI-INT16 Disp32(00f8ffff)
	vpdpwsuds	%ymm4, %ymm5, %ymm6	 #AVX-VNNI-INT16
	vpdpwsuds	%xmm4, %xmm5, %xmm6	 #AVX-VNNI-INT16
	vpdpwsuds	0x10000000(%esp, %esi, 8), %ymm5, %ymm6	 #AVX-VNNI-INT16
	vpdpwsuds	(%ecx), %ymm5, %ymm6	 #AVX-VNNI-INT16
	vpdpwsuds	4064(%ecx), %ymm5, %ymm6	 #AVX-VNNI-INT16 Disp32(e00f0000)
	vpdpwsuds	-4096(%edx), %ymm5, %ymm6	 #AVX-VNNI-INT16 Disp32(00f0ffff)
	vpdpwsuds	0x10000000(%esp, %esi, 8), %xmm5, %xmm6	 #AVX-VNNI-INT16
	vpdpwsuds	(%ecx), %xmm5, %xmm6	 #AVX-VNNI-INT16
	vpdpwsuds	2032(%ecx), %xmm5, %xmm6	 #AVX-VNNI-INT16 Disp32(f0070000)
	vpdpwsuds	-2048(%edx), %xmm5, %xmm6	 #AVX-VNNI-INT16 Disp32(00f8ffff)
	vpdpwusd	%ymm4, %ymm5, %ymm6	 #AVX-VNNI-INT16
	vpdpwusd	%xmm4, %xmm5, %xmm6	 #AVX-VNNI-INT16
	vpdpwusd	0x10000000(%esp, %esi, 8), %ymm5, %ymm6	 #AVX-VNNI-INT16
	vpdpwusd	(%ecx), %ymm5, %ymm6	 #AVX-VNNI-INT16
	vpdpwusd	4064(%ecx), %ymm5, %ymm6	 #AVX-VNNI-INT16 Disp32(e00f0000)
	vpdpwusd	-4096(%edx), %ymm5, %ymm6	 #AVX-VNNI-INT16 Disp32(00f0ffff)
	vpdpwusd	0x10000000(%esp, %esi, 8), %xmm5, %xmm6	 #AVX-VNNI-INT16
	vpdpwusd	(%ecx), %xmm5, %xmm6	 #AVX-VNNI-INT16
	vpdpwusd	2032(%ecx), %xmm5, %xmm6	 #AVX-VNNI-INT16 Disp32(f0070000)
	vpdpwusd	-2048(%edx), %xmm5, %xmm6	 #AVX-VNNI-INT16 Disp32(00f8ffff)
	vpdpwusds	%ymm4, %ymm5, %ymm6	 #AVX-VNNI-INT16
	vpdpwusds	%xmm4, %xmm5, %xmm6	 #AVX-VNNI-INT16
	vpdpwusds	0x10000000(%esp, %esi, 8), %ymm5, %ymm6	 #AVX-VNNI-INT16
	vpdpwusds	(%ecx), %ymm5, %ymm6	 #AVX-VNNI-INT16
	vpdpwusds	4064(%ecx), %ymm5, %ymm6	 #AVX-VNNI-INT16 Disp32(e00f0000)
	vpdpwusds	-4096(%edx), %ymm5, %ymm6	 #AVX-VNNI-INT16 Disp32(00f0ffff)
	vpdpwusds	0x10000000(%esp, %esi, 8), %xmm5, %xmm6	 #AVX-VNNI-INT16
	vpdpwusds	(%ecx), %xmm5, %xmm6	 #AVX-VNNI-INT16
	vpdpwusds	2032(%ecx), %xmm5, %xmm6	 #AVX-VNNI-INT16 Disp32(f0070000)
	vpdpwusds	-2048(%edx), %xmm5, %xmm6	 #AVX-VNNI-INT16 Disp32(00f8ffff)
	vpdpwuud	%ymm4, %ymm5, %ymm6	 #AVX-VNNI-INT16
	vpdpwuud	%xmm4, %xmm5, %xmm6	 #AVX-VNNI-INT16
	vpdpwuud	0x10000000(%esp, %esi, 8), %ymm5, %ymm6	 #AVX-VNNI-INT16
	vpdpwuud	(%ecx), %ymm5, %ymm6	 #AVX-VNNI-INT16
	vpdpwuud	4064(%ecx), %ymm5, %ymm6	 #AVX-VNNI-INT16 Disp32(e00f0000)
	vpdpwuud	-4096(%edx), %ymm5, %ymm6	 #AVX-VNNI-INT16 Disp32(00f0ffff)
	vpdpwuud	0x10000000(%esp, %esi, 8), %xmm5, %xmm6	 #AVX-VNNI-INT16
	vpdpwuud	(%ecx), %xmm5, %xmm6	 #AVX-VNNI-INT16
	vpdpwuud	2032(%ecx), %xmm5, %xmm6	 #AVX-VNNI-INT16 Disp32(f0070000)
	vpdpwuud	-2048(%edx), %xmm5, %xmm6	 #AVX-VNNI-INT16 Disp32(00f8ffff)
	vpdpwuuds	%ymm4, %ymm5, %ymm6	 #AVX-VNNI-INT16
	vpdpwuuds	%xmm4, %xmm5, %xmm6	 #AVX-VNNI-INT16
	vpdpwuuds	0x10000000(%esp, %esi, 8), %ymm5, %ymm6	 #AVX-VNNI-INT16
	vpdpwuuds	(%ecx), %ymm5, %ymm6	 #AVX-VNNI-INT16
	vpdpwuuds	4064(%ecx), %ymm5, %ymm6	 #AVX-VNNI-INT16 Disp32(e00f0000)
	vpdpwuuds	-4096(%edx), %ymm5, %ymm6	 #AVX-VNNI-INT16 Disp32(00f0ffff)
	vpdpwuuds	0x10000000(%esp, %esi, 8), %xmm5, %xmm6	 #AVX-VNNI-INT16
	vpdpwuuds	(%ecx), %xmm5, %xmm6	 #AVX-VNNI-INT16
	vpdpwuuds	2032(%ecx), %xmm5, %xmm6	 #AVX-VNNI-INT16 Disp32(f0070000)
	vpdpwuuds	-2048(%edx), %xmm5, %xmm6	 #AVX-VNNI-INT16 Disp32(00f8ffff)

	.intel_syntax noprefix
	vpdpwsud	ymm6, ymm5, ymm4	 #AVX-VNNI-INT16
	vpdpwsud	xmm6, xmm5, xmm4	 #AVX-VNNI-INT16
	vpdpwsud	ymm6, ymm5, YMMWORD PTR [esp+esi*8+0x10000000]	 #AVX-VNNI-INT16
	vpdpwsud	ymm6, ymm5, YMMWORD PTR [ecx]	 #AVX-VNNI-INT16
	vpdpwsud	ymm6, ymm5, YMMWORD PTR [ecx+4064]	 #AVX-VNNI-INT16 Disp32(e00f0000)
	vpdpwsud	ymm6, ymm5, YMMWORD PTR [edx-4096]	 #AVX-VNNI-INT16 Disp32(00f0ffff)
	vpdpwsud	xmm6, xmm5, XMMWORD PTR [esp+esi*8+0x10000000]	 #AVX-VNNI-INT16
	vpdpwsud	xmm6, xmm5, XMMWORD PTR [ecx]	 #AVX-VNNI-INT16
	vpdpwsud	xmm6, xmm5, XMMWORD PTR [ecx+2032]	 #AVX-VNNI-INT16 Disp32(f0070000)
	vpdpwsud	xmm6, xmm5, XMMWORD PTR [edx-2048]	 #AVX-VNNI-INT16 Disp32(00f8ffff)
	vpdpwsuds	ymm6, ymm5, ymm4	 #AVX-VNNI-INT16
	vpdpwsuds	xmm6, xmm5, xmm4	 #AVX-VNNI-INT16
	vpdpwsuds	ymm6, ymm5, YMMWORD PTR [esp+esi*8+0x10000000]	 #AVX-VNNI-INT16
	vpdpwsuds	ymm6, ymm5, YMMWORD PTR [ecx]	 #AVX-VNNI-INT16
	vpdpwsuds	ymm6, ymm5, YMMWORD PTR [ecx+4064]	 #AVX-VNNI-INT16 Disp32(e00f0000)
	vpdpwsuds	ymm6, ymm5, YMMWORD PTR [edx-4096]	 #AVX-VNNI-INT16 Disp32(00f0ffff)
	vpdpwsuds	xmm6, xmm5, XMMWORD PTR [esp+esi*8+0x10000000]	 #AVX-VNNI-INT16
	vpdpwsuds	xmm6, xmm5, XMMWORD PTR [ecx]	 #AVX-VNNI-INT16
	vpdpwsuds	xmm6, xmm5, XMMWORD PTR [ecx+2032]	 #AVX-VNNI-INT16 Disp32(f0070000)
	vpdpwsuds	xmm6, xmm5, XMMWORD PTR [edx-2048]	 #AVX-VNNI-INT16 Disp32(00f8ffff)
	vpdpwusd	ymm6, ymm5, ymm4	 #AVX-VNNI-INT16
	vpdpwusd	xmm6, xmm5, xmm4	 #AVX-VNNI-INT16
	vpdpwusd	ymm6, ymm5, YMMWORD PTR [esp+esi*8+0x10000000]	 #AVX-VNNI-INT16
	vpdpwusd	ymm6, ymm5, YMMWORD PTR [ecx]	 #AVX-VNNI-INT16
	vpdpwusd	ymm6, ymm5, YMMWORD PTR [ecx+4064]	 #AVX-VNNI-INT16 Disp32(e00f0000)
	vpdpwusd	ymm6, ymm5, YMMWORD PTR [edx-4096]	 #AVX-VNNI-INT16 Disp32(00f0ffff)
	vpdpwusd	xmm6, xmm5, XMMWORD PTR [esp+esi*8+0x10000000]	 #AVX-VNNI-INT16
	vpdpwusd	xmm6, xmm5, XMMWORD PTR [ecx]	 #AVX-VNNI-INT16
	vpdpwusd	xmm6, xmm5, XMMWORD PTR [ecx+2032]	 #AVX-VNNI-INT16 Disp32(f0070000)
	vpdpwusd	xmm6, xmm5, XMMWORD PTR [edx-2048]	 #AVX-VNNI-INT16 Disp32(00f8ffff)
	vpdpwusds	ymm6, ymm5, ymm4	 #AVX-VNNI-INT16
	vpdpwusds	xmm6, xmm5, xmm4	 #AVX-VNNI-INT16
	vpdpwusds	ymm6, ymm5, YMMWORD PTR [esp+esi*8+0x10000000]	 #AVX-VNNI-INT16
	vpdpwusds	ymm6, ymm5, YMMWORD PTR [ecx]	 #AVX-VNNI-INT16
	vpdpwusds	ymm6, ymm5, YMMWORD PTR [ecx+4064]	 #AVX-VNNI-INT16 Disp32(e00f0000)
	vpdpwusds	ymm6, ymm5, YMMWORD PTR [edx-4096]	 #AVX-VNNI-INT16 Disp32(00f0ffff)
	vpdpwusds	xmm6, xmm5, XMMWORD PTR [esp+esi*8+0x10000000]	 #AVX-VNNI-INT16
	vpdpwusds	xmm6, xmm5, XMMWORD PTR [ecx]	 #AVX-VNNI-INT16
	vpdpwusds	xmm6, xmm5, XMMWORD PTR [ecx+2032]	 #AVX-VNNI-INT16 Disp32(f0070000)
	vpdpwusds	xmm6, xmm5, XMMWORD PTR [edx-2048]	 #AVX-VNNI-INT16 Disp32(00f8ffff)
	vpdpwuud	ymm6, ymm5, ymm4	 #AVX-VNNI-INT16
	vpdpwuud	xmm6, xmm5, xmm4	 #AVX-VNNI-INT16
	vpdpwuud	ymm6, ymm5, YMMWORD PTR [esp+esi*8+0x10000000]	 #AVX-VNNI-INT16
	vpdpwuud	ymm6, ymm5, YMMWORD PTR [ecx]	 #AVX-VNNI-INT16
	vpdpwuud	ymm6, ymm5, YMMWORD PTR [ecx+4064]	 #AVX-VNNI-INT16 Disp32(e00f0000)
	vpdpwuud	ymm6, ymm5, YMMWORD PTR [edx-4096]	 #AVX-VNNI-INT16 Disp32(00f0ffff)
	vpdpwuud	xmm6, xmm5, XMMWORD PTR [esp+esi*8+0x10000000]	 #AVX-VNNI-INT16
	vpdpwuud	xmm6, xmm5, XMMWORD PTR [ecx]	 #AVX-VNNI-INT16
	vpdpwuud	xmm6, xmm5, XMMWORD PTR [ecx+2032]	 #AVX-VNNI-INT16 Disp32(f0070000)
	vpdpwuud	xmm6, xmm5, XMMWORD PTR [edx-2048]	 #AVX-VNNI-INT16 Disp32(00f8ffff)
	vpdpwuuds	ymm6, ymm5, ymm4	 #AVX-VNNI-INT16
	vpdpwuuds	xmm6, xmm5, xmm4	 #AVX-VNNI-INT16
	vpdpwuuds	ymm6, ymm5, YMMWORD PTR [esp+esi*8+0x10000000]	 #AVX-VNNI-INT16
	vpdpwuuds	ymm6, ymm5, YMMWORD PTR [ecx]	 #AVX-VNNI-INT16
	vpdpwuuds	ymm6, ymm5, YMMWORD PTR [ecx+4064]	 #AVX-VNNI-INT16 Disp32(e00f0000)
	vpdpwuuds	ymm6, ymm5, YMMWORD PTR [edx-4096]	 #AVX-VNNI-INT16 Disp32(00f0ffff)
	vpdpwuuds	xmm6, xmm5, XMMWORD PTR [esp+esi*8+0x10000000]	 #AVX-VNNI-INT16
	vpdpwuuds	xmm6, xmm5, XMMWORD PTR [ecx]	 #AVX-VNNI-INT16
	vpdpwuuds	xmm6, xmm5, XMMWORD PTR [ecx+2032]	 #AVX-VNNI-INT16 Disp32(f0070000)
	vpdpwuuds	xmm6, xmm5, XMMWORD PTR [edx-2048]	 #AVX-VNNI-INT16 Disp32(00f8ffff)
