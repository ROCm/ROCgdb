	# noexact by default.
	lw	a0, 0(a0)
	c.lw	a0, 0(a0)

	.option	exact
	lw	a0, 0(a0)
	c.lw	a0, 0(a0)

	.option	noexact
	lw	a0, 0(a0)
	c.lw	a0, 0(a0)
