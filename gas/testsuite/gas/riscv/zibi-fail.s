foo:
	beqi t0, -2, foo
	beqi t4, 32, foo
	beqi t4, 34, foo
	beqi t2, 0, foo
	bnei t0, -3, foo
	bnei t4, 32, foo
	bnei t2, 33, foo
	bnei t3, 0, foo
