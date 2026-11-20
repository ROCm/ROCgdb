# With -mno-relax, a branch to a local label in another section still
# needs a relocation.  One in the same section is resolved.

	.option exact
	.text
	beq	a0, a1, .Lother
	jal	.Lother
	c.beqz	a0, .Lother
	c.j	.Lother
	beq	a0, a1, .Lsame
	jal	.Lsame
	c.beqz	a0, .Lsame
	c.j	.Lsame
.Lsame:
	ret

	.section .text.other, "ax"
	nop
.Lother:
	ret
