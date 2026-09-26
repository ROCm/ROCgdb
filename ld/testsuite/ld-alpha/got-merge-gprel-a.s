/* 5000 got entries (40000 bytes, the first .got subsection) against a
   symbol too far from any gp for relaxation to remove them.  */

	.text
	.globl	afunc
	.ent	afunc
afunc:
	ldgp	$29,0($27)
	.prologue 1
	.irpc	w,01234
	.irpc	x,0123456789
	.irpc	y,0123456789
	.irpc	z,0123456789
	ldq	$16,far+(1\w\x\y\z-10000)*8($29)	!literal
	.endr
	.endr
	.endr
	.endr
	ret	$31,($26),1
	.end	afunc
