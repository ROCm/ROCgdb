/* 3000 more got entries against far, and 1000 against near in .sbss
   just past the .got.  With got-merge-gprel-a.s this exceeds
   MAX_GOT_SIZE, so this object gets its own .got subsection.  Once the
   near loads are relaxed to GPREL16 the two would fit in one, but
   merging then would move this object's gp down by 40000 bytes.  */

	.text
	.globl	_start
	.ent	_start
_start:
	ldgp	$29,0($27)
	.prologue 1
	.irpc	w,567
	.irpc	x,0123456789
	.irpc	y,0123456789
	.irpc	z,0123456789
	ldq	$16,far+(1\w\x\y\z-10000)*8($29)	!literal
	.endr
	.endr
	.endr
	.endr
	.irpc	x,0123456789
	.irpc	y,0123456789
	.irpc	z,0123456789
	ldq	$16,near+(1\x\y\z-1000)*8($29)	!literal
	.endr
	.endr
	.endr
	ret	$31,($26),1
	.end	_start

	.section .sbss,"aw",@nobits
	.globl	near
near:
	.skip	8000

	.bss
	.skip	0x200000
	.globl	far
far:
	.skip	64000
