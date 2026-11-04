fmmla z0.h, z0.h, z0.h
fmmla z31.h, z0.h, z0.h
fmmla z0.h, z31.h, z0.h
fmmla z0.h, z0.h, z31.h
movprfx	z31, z15
fmmla z31.h, z0.h, z0.h
movprfx	z0, z15
fmmla z0.h, z1.h, z2.h
