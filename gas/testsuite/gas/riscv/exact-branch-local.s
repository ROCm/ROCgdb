.ifdef exact_mode
	.option	exact
.else
	.option	noexact
.endif

	## Conditional Branches

	# Zero offset.
	beq	a0, a1, .
	bne	a0, a1, .
	blt	a0, a1, .
	bge	a0, a1, .
	bltu	a0, a1, .
	bgeu	a0, a1, .

	# Zero offset (aliases).
	bgt	a0, a1, .
	ble	a0, a1, .
	bgtu	a0, a1, .
	bleu	a0, a1, .
	beqz	a0, .
	bnez	a0, .
	bltz	a0, .
	bgtz	a0, .
	blez	a0, .
	bgez	a0, .

	# Offsets that c.b{ne,eq}z can represent.
	beqz	a0, .+0x0fe
	beqz	a0, .-0x100
	bnez	a0, .+0x0fe
	bnez	a0, .-0x100

	# Offsets that c.b{ne,eq}z cannot represent but single b{ne,eq}z can.
	beqz	a0, .+0x0ffe
	beqz	a0, .-0x1000
	bnez	a0, .+0x0ffe
	bnez	a0, .-0x1000

	# Offsets that single conditional branch instruction can represent.
	beq	a0, a1, .+0x0ffe
	beq	a0, a1, .-0x1000
	bne	a0, a1, .+0x0ffe
	bne	a0, a1, .-0x1000
	blt	a0, a1, .+0x0ffe
	blt	a0, a1, .-0x1000
	bge	a0, a1, .+0x0ffe
	bge	a0, a1, .-0x1000
	bltu	a0, a1, .+0x0ffe
	bltu	a0, a1, .-0x1000
	bgeu	a0, a1, .+0x0ffe
	bgeu	a0, a1, .-0x1000
	bgt	a0, a1, .+0x0ffe
	bgt	a0, a1, .-0x1000
	ble	a0, a1, .+0x0ffe
	ble	a0, a1, .-0x1000
	bgtu	a0, a1, .+0x0ffe
	bgtu	a0, a1, .-0x1000
	bleu	a0, a1, .+0x0ffe
	bleu	a0, a1, .-0x1000
	beqz	a0, .+0x0ffe
	beqz	a0, .-0x1000
	bnez	a0, .+0x0ffe
	bnez	a0, .-0x1000
	bltz	a0, .+0x0ffe
	bltz	a0, .-0x1000
	bgtz	a0, .+0x0ffe
	bgtz	a0, .-0x1000
	blez	a0, .+0x0ffe
	blez	a0, .-0x1000
	bgez	a0, .+0x0ffe
	bgez	a0, .-0x1000

	## Unconditional Jumps (normal and compressed)

	# Zero offset.
	jal	a0, .
	c.j	.
	c.jal	.	# RV32C only

	# Offsets that single jump instruction can represent.
	jal	a0, .+0x0ffffe
	jal	a0, .-0x100000
	c.j	.+0x7fe
	c.j	.-0x800
	c.jal	.+0x7fe	# RV32C only
	c.jal	.-0x800	# RV32C only

.ifdef long_branch
	# Offsets that single conditional branch instruction cannot represent.
	# (should fail on the exact mode)
	beq	a0, a1, .+0x0ffe+2
	beq	a0, a1, .-0x1000-2
	bne	a0, a1, .+0x0ffe+2
	bne	a0, a1, .-0x1000-2
	blt	a0, a1, .+0x0ffe+2
	blt	a0, a1, .-0x1000-2
	bge	a0, a1, .+0x0ffe+2
	bge	a0, a1, .-0x1000-2
	bltu	a0, a1, .+0x0ffe+2
	bltu	a0, a1, .-0x1000-2
	bgeu	a0, a1, .+0x0ffe+2
	bgeu	a0, a1, .-0x1000-2
	bgt	a0, a1, .+0x0ffe+2
	bgt	a0, a1, .-0x1000-2
	ble	a0, a1, .+0x0ffe+2
	ble	a0, a1, .-0x1000-2
	bgtu	a0, a1, .+0x0ffe+2
	bgtu	a0, a1, .-0x1000-2
	bleu	a0, a1, .+0x0ffe+2
	bleu	a0, a1, .-0x1000-2
	beqz	a0, .+0x0ffe+2
	beqz	a0, .-0x1000-2
	bnez	a0, .+0x0ffe+2
	bnez	a0, .-0x1000-2
	bltz	a0, .+0x0ffe+2
	bltz	a0, .-0x1000-2
	bgtz	a0, .+0x0ffe+2
	bgtz	a0, .-0x1000-2
	blez	a0, .+0x0ffe+2
	blez	a0, .-0x1000-2
	bgez	a0, .+0x0ffe+2
	bgez	a0, .-0x1000-2

	# Offsets that single instruction cannot represent.
	# Fail on the exact mode, either fallback or linker error on the non-exact mode.
	c.beqz	a0, .+0x0fe+2
	c.beqz	a0, .-0x100-2
	c.bnez	a0, .+0x0fe+2
	c.bnez	a0, .-0x100-2
	jal	a0, .+0x0ffffe+2	# results in a linker error on the non-exact mode
	jal	a0, .-0x100000-2	# results in a linker error on the non-exact mode
	c.j	.+0x7fe+2
	c.j	.-0x800-2
	c.jal	.+0x7fe+2	# RV32C only
	c.jal	.-0x800-2	# RV32C only
.endif
