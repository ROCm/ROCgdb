bfmmla z0.h, z0.h, z0.h
bfmmla z31.h, z0.h, z0.h
bfmmla z0.h, z31.h, z0.h
bfmmla z0.h, z0.h, z31.h
movprfx	z31, z15
bfmmla z31.h, z0.h, z0.h
movprfx	z0, z15
bfmmla z0.h, z1.h, z2.h
