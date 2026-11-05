	.text
	# vadd.vv v1, v2, v3, under the "$xrv" symbol gas emits here.
	.insn 4, 0x022180d7
	# The same word after a bare "$x", which returns to the default ISA.
$x:
	.insn 4, 0x022180d7
