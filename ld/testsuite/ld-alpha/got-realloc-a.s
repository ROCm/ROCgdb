/* 8000 got entries against preemptible symbols, so relaxation cannot
   remove any of them.  That is 64000 bytes, just under MAX_GOT_SIZE, so
   this object keeps its own got subsection.  */

	.text
	.globl	afunc
	.ent	afunc
afunc:
	ldgp	$29,0($27)
	.prologue 1
	.irpc	w,01234567
	.irpc	x,0123456789
	.irpc	y,0123456789
	.irpc	z,0123456789
	ldq	$1,g\w\x\y\z($29)	!literal
	.endr
	.endr
	.endr
	.endr
	ret	$31,($26),1
	.end	afunc
