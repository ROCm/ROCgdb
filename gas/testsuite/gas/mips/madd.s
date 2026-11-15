# Source file used to test the integer multiply-add hazard.

	.text
	.ent	foo
foo:
	madd	$4, $5
	div	$0, $6, $7
	madd	$2, $4, $5
	div	$0, $6, $7
	maddu	$4, $5
	divu	$0, $6, $7
	maddu	$2, $4, $5
	divu	$0, $6, $7
	.end	foo
