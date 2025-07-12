.ifdef exact_mode
	.option	exact
.else
	.option	noexact
.endif
	.extern	ext

	## Conditional Branches

	# Basic instructions
	beq	a0, a1, ext
	bne	a0, a1, ext
	blt	a0, a1, ext
	bge	a0, a1, ext
	bltu	a0, a1, ext
	bgeu	a0, a1, ext

	# Compressed instructions
	c.beqz	a0, ext
	c.bnez	a0, ext

	# Aliases
	bgt	a0, a1, ext
	ble	a0, a1, ext
	bgtu	a0, a1, ext
	bleu	a0, a1, ext
	beqz	a0, ext
	bnez	a0, ext
	bltz	a0, ext
	bgtz	a0, ext
	blez	a0, ext
	bgez	a0, ext

	c.nop

	## Unconditional Jumps (normal and compressed)

	jal	a0, ext
	c.j	ext
	c.jal	ext	# RV32C only
