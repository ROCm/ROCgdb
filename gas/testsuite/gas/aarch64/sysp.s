	.arch armv9.4-a+d128

	sysp	#0, C8, C0, #0, x0, x1
	sysp	#6, C9, C7, #7, x26, x27
	sysp	#0, C0, C0, #0, x0, x1
	sysp	#7, C0, C0, #0, x0, x1
	sysp	#0, C15, C0, #0, x0, x1
	sysp	#0, C0, C15, #0, x0, x1
	sysp	#0, C0, C0, #7, x0, x1
	sysp	#0, C0, C0, #0, xzr, xzr
	sysp	#0, C0, C0, #0
