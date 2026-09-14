/* 250 got entries, of which relaxation removes the 150 that are only used
   by a call to a local function.  Adding this object's 250 entries to the
   8000 in got-realloc-a.s exceeds MAX_GOT_SIZE, so the two subsections
   cannot be merged when they are first sized.  Once relaxation has dropped
   the 150 they fit, and re-merging grows the first subsection from 64000
   bytes to 64800.  */

	.macro	mkcall name, seq
	ldq	$27,\name($29)		!literal!\seq
	jsr	$26,($27),\name		!lituse_jsr!\seq
	.endm

	.text

	.irpc	x,012
	.irpc	y,0123456789
	.irpc	z,01234
	.ent	lf\x\y\z
lf\x\y\z:
	ldgp	$29,0($27)
	.prologue 1
	ret	$31,($26),1
	.end	lf\x\y\z
	.endr
	.endr
	.endr

	.globl	bfunc
	.ent	bfunc
bfunc:
	ldgp	$29,0($27)
	.prologue 1
	.irpc	x,012
	.irpc	y,0123456789
	.irpc	z,01234
	mkcall	lf\x\y\z, 1\x\y\z
	.endr
	.endr
	.endr
	.irpc	x,0123456789
	.irpc	y,0123456789
	ldq	$1,h\x\y($29)		!literal
	.endr
	.endr
	ret	$31,($26),1
	.end	bfunc
