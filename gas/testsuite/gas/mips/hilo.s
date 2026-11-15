# Source file used to test the moves to/from HI/LO registers.

	.text
	.ent	foo
foo:
	mflo	$2
	mtlo	$3
	mfhi	$4
	mthi	$5
	mflo	$6
	mthi	$7
	mfhi	$16
	mtlo	$17
	mflo	$18
	.end	foo
