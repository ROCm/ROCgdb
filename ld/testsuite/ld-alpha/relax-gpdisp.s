/* --relax turns the LITERAL+LITUSE_JSR call into a bsr straight to
   foo+8, since foo shares our gp and starts with an ldgp.  The GPDISP
   ldah/lda pair after the call must survive: foo is free to return with
   some other gp in $29.  */

	.text

	.globl	_start
	.ent	_start
_start:
	ldgp	$29,0($27)
	.prologue 1
	ldq	$27,foo($29)		!literal!1
	jsr	$26,($27),foo		!lituse_jsr!1
	ldah	$29,0($26)		!gpdisp!2
	lda	$29,0($29)		!gpdisp!2
	ret	$31,($26),1
	.end	_start

	.globl	foo
	.ent	foo
foo:
	ldgp	$29,0($27)
	.prologue 1
	ret	$31,($26),1
	.end	foo
