	add	za.d[w8, 0], { z0.d - z1.d }
	add	za.d[w8, 0, vgx2], { z0.d - z1.d }
	ADD	ZA.d[W8, 0, VGx2], { Z0.d - Z1.d }
	ADD	ZA.D[W8, 0, VGX2], { Z0.D - Z1.D }
	add	za.d[w11, 0], { z0.d - z1.d }
	add	za.d[w8, 7], { z0.d - z1.d }
	add	za.d[w8, 0], { z30.d - z31.d }
	add	za.d[w10, 3], { z10.d - z11.d }

	add	za.d[w8, 0], { z0.d - z3.d }
	add	za.d[w8, 0, vgx4], { z0.d - z3.d }
	ADD	ZA.d[W8, 0, VGx4], { Z0.d - Z3.d }
	ADD	ZA.D[W8, 0, VGX4], { Z0.D - Z3.D }
	add	za.d[w11, 0], { z0.d - z3.d }
	add	za.d[w8, 7], { z0.d - z3.d }
	add	za.d[w8, 0], { z28.d - z31.d }
	add	za.d[w11, 1], { z12.d - z15.d }

	add	za.d[w8, 0], { z0.d - z1.d }, z0.d
	add	za.d[w8, 0, vgx2], { z0.d - z1.d }, z0.d
	ADD	ZA.d[W8, 0, VGx2], { Z0.d - Z1.d }, Z0.d
	ADD	ZA.D[W8, 0, VGX2], { Z0.D - Z1.D }, Z0.D
	add	za.d[w11, 0], { z0.d - z1.d }, z0.d
	add	za.d[w8, 7], { z0.d - z1.d }, z0.d
	add	za.d[w8, 0], { z30.d - z31.d }, z0.d
	add	za.d[w8, 0], { z31.d, z0.d }, z0.d
	add	za.d[w8, 0], { z31.d - z0.d }, z0.d
	add	za.d[w8, 0], { z0.d - z1.d }, z15.d
	add	za.d[w9, 5], { z9.d - z10.d }, z6.d

	add	za.d[w8, 0], { z0.d - z3.d }, z0.d
	add	za.d[w8, 0, vgx4], { z0.d - z3.d }, z0.d
	ADD	ZA.d[W8, 0, VGx4], { Z0.d - Z3.d }, Z0.d
	ADD	ZA.D[W8, 0, VGX4], { Z0.D - Z3.D }, Z0.D
	add	za.d[w11, 0], { z0.d - z3.d }, z0.d
	add	za.d[w8, 7], { z0.d - z3.d }, z0.d
	add	za.d[w8, 0], { z28.d - z31.d }, z0.d
	add	za.d[w8, 0], { z31.d, z0.d, z1.d, z2.d }, z0.d
	add	za.d[w8, 0], { z31.d - z2.d }, z0.d
	add	za.d[w8, 0], { z0.d - z3.d }, z15.d
	add	za.d[w11, 2], { z23.d - z26.d }, z13.d

	add	za.d[w8, 0], { z0.d - z1.d }, { z0.d - z1.d }
	add	za.d[w8, 0, vgx2], { z0.d - z1.d }, { z0.d - z1.d }
	ADD	ZA.d[W8, 0, VGx2], { Z0.d - Z1.d }, { Z0.d - Z1.d }
	ADD	ZA.D[W8, 0, VGX2], { Z0.D - Z1.D }, { Z0.D - Z1.D }
	add	za.d[w11, 0], { z0.d - z1.d }, { z0.d - z1.d }
	add	za.d[w8, 7], { z0.d - z1.d }, { z0.d - z1.d }
	add	za.d[w8, 0], { z30.d - z31.d }, { z0.d - z1.d }
	add	za.d[w8, 0], { z0.d - z1.d }, { z30.d - z31.d }
	add	za.d[w10, 1], { z22.d - z23.d }, { z18.d - z19.d }

	add	za.d[w8, 0], { z0.d - z3.d }, { z0.d - z3.d }
	add	za.d[w8, 0, vgx4], { z0.d - z3.d }, { z0.d - z3.d }
	ADD	ZA.d[W8, 0, VGx4], { Z0.d - Z3.d }, { Z0.d - Z3.d }
	ADD	ZA.D[W8, 0, VGX4], { Z0.D - Z3.D }, { Z0.D - Z3.D }
	add	za.d[w11, 0], { z0.d - z3.d }, { z0.d - z3.d }
	add	za.d[w8, 7], { z0.d - z3.d }, { z0.d - z3.d }
	add	za.d[w8, 0], { z28.d - z31.d }, { z0.d - z3.d }
	add	za.d[w8, 0], { z0.d - z3.d }, { z28.d - z31.d }
	add	za.d[w11, 3], { z16.d - z19.d }, { z24.d - z27.d }
