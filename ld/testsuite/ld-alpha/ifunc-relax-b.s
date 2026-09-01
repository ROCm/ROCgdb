/* 251 got entries, of which relaxation removes the 150 that are only used
   by a call to a local function.  Adding this object's entries to the 8001
   in ifunc-relax-a.s exceeds MAX_GOT_SIZE, so the two subsections cannot be
   merged when they are first sized.  Once relaxation has dropped the 150
   they fit, and the two entries for the IFUNC merge into one, which is one
   fewer R_ALPHA_IRELATIVE than the first sizing reserved.  */

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
	ldq	$1,global_ifunc($29)	!literal
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
