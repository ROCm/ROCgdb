target:
	jr	0(a2)
	jr	(a2)
	jr	a2, 0
	jalr	0(a2)
	jalr	(a2)
	jalr	a2, 0
	jalr	x0, a2
	jalr	x0, 0(a2)
	jalr	x0, (a2)
	jalr	x0, a2, 0
	jalr	ra, a2
	jalr	ra, 0(a2)
	jalr	ra, (a2)
	jalr	ra, a2, 0
	# Not compressible.
	jalr	x0, 4(a2)
	jalr	x0, 0(x0)
	jalr	a0, 0(a2)
	.option norvc
	jalr	x0, 0(a2)
	jalr	ra, 0(a2)
	jr	(a2)
	.option rvc
	.option exact
	jalr	x0, 0(a2)
	jalr	ra, 0(a2)
	jr	(a2)
