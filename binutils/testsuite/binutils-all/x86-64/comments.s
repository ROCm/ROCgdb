	.text
	.global _start
_start:
	movq	$_start, fptr(%rip)
	xor	%eax, %eax
	ret

	.data
fptr:	.quad	-1
