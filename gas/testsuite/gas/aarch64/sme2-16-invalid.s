	sdot	0, { z0.h - z1.h }, z0.h[0]
	sdot	za.s[w8, 0], 0, z0.h[0]
	sdot	za.s[w8, 0], { z0.h - z1.h }, 0

	sdot	za.h[w8, 0], z0.h, z0.h
	sdot	za.h[w8, 0], { z0.h - z1.h }, z0.h

	sdot	za.s[w7, 0], { z0.h - z1.h }, z0.h[0]
	sdot	za.s[w12, 0], { z0.h - z1.h }, z0.h[0]
	sdot	za.s[w8, -1], { z0.h - z1.h }, z0.h[0]
	sdot	za.s[w8, 8], { z0.h - z1.h }, z0.h[0]
	sdot	za.s[w8, 0], { z0.h - z2.h }, z0.h[0]
	sdot	za.s[w8, 0], { z1.h - z2.h }, z0.h[0]
	sdot	za.s[w8, 0], { z0.h - z1.h }, z0.h[-1]
	sdot	za.s[w8, 0], { z0.h - z1.h }, z0.h[4]
	sdot	za.s[w8, 0], { z0.h - z1.h }, z16.h[0]

	sdot	za.s[w7, 0], { z0.h - z3.h }, z0.h[0]
	sdot	za.s[w12, 0], { z0.h - z3.h }, z0.h[0]
	sdot	za.s[w8, -1], { z0.h - z3.h }, z0.h[0]
	sdot	za.s[w8, 8], { z0.h - z3.h }, z0.h[0]
	sdot	za.s[w8, 0], { z1.h - z4.h }, z0.h[0]
	sdot	za.s[w8, 0], { z2.h - z5.h }, z0.h[0]
	sdot	za.s[w8, 0], { z3.h - z6.h }, z0.h[0]
	sdot	za.s[w8, 0], { z0.h - z3.h }, z0.h[-1]
	sdot	za.s[w8, 0], { z0.h - z3.h }, z0.h[4]
	sdot	za.s[w8, 0], { z0.h - z3.h }, z16.h[0]

	sdot	za.s[w8, 0, vgx4], { z0.h - z1.h }, z0.h[0]
	sdot	za.s[w8, 0, vgx2], { z0.h - z3.h }, z0.h[0]
	sdot	za[w8, 0], { z0.h - z1.h }, z0.h[0]
	sdot	za.s[w8, 0], { z0 - z1 }, z0.h[0]
	sdot	za.s[w8, 0], { z0.h - z1.h }, z0[0]
	sdot	za.h[w8, 0], { z0.h - z1.h }, z0.h[0]
	sdot	za.h[w8, 0], { z0.s - z1.s }, z0.s[0]

	sdot	za.s[w8, 0], { z0.h - z2.h }, z0.h
	sdot	za.s[w8, 0], { z0.h - z4.h }, z0.h
	sdot	za.s[w8, 0], { z0.h, z1.h, z2.h }, z0.h
	sdot	za.s[w8, 0], { z0.h, z1.h, z5.h }, z0.h

	sdot	za.s[w8, 0, vgx4], { z0.h - z1.h }, z0.h
	sdot	za.s[w8, 0, vgx2], { z0.h - z3.h }, z0.h
	sdot	za[w8, 0], { z0.h - z1.h }, z0.h
	sdot	za.s[w8, 0], { z0 - z1 }, z0.h
	sdot	za.s[w8, 0], { z0.h - z1.h }, z0
	sdot	za[w8, 0], { z0.h - z1.h }, z0

	sdot	za.s[w7, 0], { z0.h - z1.h }, { z0.h - z1.h }
	sdot	za.s[w12, 0], { z0.h - z1.h }, { z0.h - z1.h }
	sdot	za.s[w8, -1], { z0.h - z1.h }, { z0.h - z1.h }
	sdot	za.s[w8, 8], { z0.h - z1.h }, { z0.h - z1.h }
	sdot	za.s[w8, 0], { z1.h - z2.h }, { z0.h - z1.h }
	sdot	za.s[w8, 0], { z0.h - z1.h }, { z15.h - z16.h }
	sdot	za.s[w8, 0], { z0.h - z1.h }, { z31.h, z0.h }

	sdot	za.s[w7, 0], { z0.h - z3.h }, { z0.h - z3.h }
	sdot	za.s[w12, 0], { z0.h - z3.h }, { z0.h - z3.h }
	sdot	za.s[w8, -1], { z0.h - z3.h }, { z0.h - z3.h }
	sdot	za.s[w8, 8], { z0.h - z3.h }, { z0.h - z3.h }
	sdot	za.s[w8, 0], { z1.h - z4.h }, { z0.h - z3.h }
	sdot	za.s[w8, 0], { z2.h - z5.h }, { z0.h - z3.h }
	sdot	za.s[w8, 0], { z3.h - z6.h }, { z0.h - z3.h }
	sdot	za.s[w8, 0], { z0.h - z3.h }, { z15.h - z18.h }
	sdot	za.s[w8, 0], { z0.h - z3.h }, { z29.h, z30.h, z31.h, z0.h }

	sdot	za.s[w8, 0], { z0.h - z2.h }, { z0.h - z1.h }
	sdot	za.s[w8, 0], { z0.h - z3.h }, { z0.h - z1.h }
	sdot	za.s[w8, 0], { z0.h - z1.h }, { z0.h - z2.h }
	sdot	za.s[w8, 0], { z0.h - z1.h }, { z0.h - z3.h }
	sdot	za.s[w8, 0], { z0.h - z1.h }, { z0.h - z4.h }

	sdot	za.s[w8, 0, vgx4], { z0.h - z1.h }, { z0.h - z3.h }
	sdot	za.s[w8, 0, vgx4], { z0.h - z3.h }, { z0.h - z1.h }
	sdot	za.s[w8, 0, vgx2], { z0.h - z1.h }, { z0.h - z3.h }
	sdot	za.s[w8, 0, vgx2], { z0.h - z3.h }, { z0.h - z1.h }
	sdot	za[w8, 0], { z0.h - z1.h }, { z0.h - z1.h }
	sdot	za[w8, 0], { z0.h - z3.h }, { z0.h - z3.h }

	sdot	za.s[w8, 0:0], { z0.h - z3.h }, { z0.h - z3.h }
	sdot	za.s[w8, 0:1], { z0.h - z3.h }, { z0.h - z3.h }
	sdot	za.s[w8, 0:2], { z0.h - z3.h }, { z0.h - z3.h }
	sdot	za.s[w8, 0:3], { z0.h - z3.h }, { z0.h - z3.h }
	sdot	za.s[w8, 1:0], { z0.h - z3.h }, { z0.h - z3.h }
	sdot	za.s[w8, foo:1], { z0.h - z3.h }, { z0.h - z3.h }
	sdot	za.s[w8, 1:foo], { z0.h - z3.h }, { z0.h - z3.h }
	sdot	za.s[w8, foo:bar], { z0.h - z3.h }, { z0.h - z3.h }
