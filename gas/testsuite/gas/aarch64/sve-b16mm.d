#as: -march=armv8-a+sve-b16mm
#objdump: -dr

.*:     file format .*

Disassembly of section \.text:

0+ <\.text>:
 *[0-9a-f]+:	64e0e000 	bfmmla	z0.h, z0.h, z0.h
 *[0-9a-f]+:	64e0e01f 	bfmmla	z31.h, z0.h, z0.h
 *[0-9a-f]+:	64e0e3e0 	bfmmla	z0.h, z31.h, z0.h
 *[0-9a-f]+:	64ffe000 	bfmmla	z0.h, z0.h, z31.h
 *[0-9a-f]+:	0420bdff 	movprfx	z31, z15
 *[0-9a-f]+:	64e0e01f 	bfmmla	z31.h, z0.h, z0.h
 *[0-9a-f]+:	0420bde0 	movprfx	z0, z15
 *[0-9a-f]+:	64e2e020 	bfmmla	z0.h, z1.h, z2.h
