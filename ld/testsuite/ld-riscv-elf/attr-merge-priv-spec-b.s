# The assembler ignores the privileged elf attributes, so write them by hand.
	.section .riscv.attributes, "", %0x70000003
	.byte	0x41
.Lvendor:
	.4byte	.Lend - .Lvendor
	.asciz	"riscv"
.Lfile:
	.byte	1
	.4byte	.Lend - .Lfile
	# Tag_RISCV_priv_spec, Tag_RISCV_priv_spec_minor.
	.byte	8, 1, 10, 10
.Lend:
