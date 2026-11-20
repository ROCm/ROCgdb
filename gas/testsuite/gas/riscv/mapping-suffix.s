# gas never emits $x.<any> or $d.<any>, so define them as labels.
	.attribute arch, "rv32ic"
	.option norvc
	.insn 4, 0x00000013
$x.0:
	.insn 2, 0x0001
	.insn 2, 0x0001
$d.1:
	.insn 4, 0x00000013
