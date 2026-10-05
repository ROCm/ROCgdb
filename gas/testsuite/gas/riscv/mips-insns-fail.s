	mips.pref	0, 0(t1)
	mips.ccmov	a0,a1,a2,a3
	mips.ehb
	mips.ldp	t3, t4, 0(t5)
