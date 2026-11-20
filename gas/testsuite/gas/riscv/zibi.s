foo:
	beqi t0, -1, foo
	beqi t4, 5, foo
	beqi t2, 15, foo
	beqi t3, 31, foo
	bnei t0, -1, foo
	bnei t4, 5, foo
	bnei t2, 15, foo
	bnei t3, 31, foo
