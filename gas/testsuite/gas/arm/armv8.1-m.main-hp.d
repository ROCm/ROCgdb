# name: Half precision instructions for 'armv8.1-m.main'
# source: armv8-2-fp16-scalar.s
# as: -march=armv8.1-m.main+fp
# objdump: -d
# skip: *-*-*coff *-*-pe *-*-wince *-*-*aout* *-*-netbsd

.*: +file format .*arm.*
Disassembly of section .text:

00000000 <label-0xc>:
   0:	ee00 1910 	vmov.f16	s0, r1
   4:	ee10 0990 	vmov.f16	r0, s1
   8:	eeb0 0900 	vmov.f16	s0, #0	@ 0x40000000  2.0

0000000c <label>:
   c:	00000ffe 	.word	0x00000ffe
  10:	ed5f 1904 	vldr.16	s3, \[pc, #-8\]	@ c <label>
  14:	ed1f 3902 	vldr.16	s6, \[pc, #-4\]	@ 14 <label\+0x8>
  18:	eddf 1902 	vldr.16	s3, \[pc, #4\]	@ 20 <label\+0x14>
  1c:	edd0 0902 	vldr.16	s1, \[r0, #4\]
  20:	ed10 1902 	vldr.16	s2, \[r0, #-4\]
  24:	ed80 3902 	vstr.16	s6, \[r0, #4\]
  28:	ed40 5902 	vstr.16	s11, \[r0, #-4\]
  2c:	eec6 298c 	vdiv.f16	s5, s13, s24
  30:	eee6 298c 	vfma.f16	s5, s13, s24
  34:	eee6 29cc 	vfms.f16	s5, s13, s24
  38:	eed6 29cc 	vfnma.f16	s5, s13, s24
  3c:	eed6 298c 	vfnms.f16	s5, s13, s24
  40:	fec6 298c 	vmaxnm.f16	s5, s13, s24
  44:	fec6 29cc 	vminnm.f16	s5, s13, s24
  48:	ee46 298c 	vmla.f16	s5, s13, s24
  4c:	ee46 29cc 	vmls.f16	s5, s13, s24
  50:	ee66 298c 	vmul.f16	s5, s13, s24
  54:	ee56 29cc 	vnmla.f16	s5, s13, s24
  58:	ee56 298c 	vnmls.f16	s5, s13, s24
  5c:	ee66 29cc 	vnmul.f16	s5, s13, s24
  60:	ee76 29cc 	vsub.f16	s5, s13, s24
  64:	eef0 29c6 	vabs.f16	s5, s12
  68:	ee72 2986 	vadd.f16	s5, s5, s12
  6c:	eef1 29c6 	vsqrt.f16	s5, s12
  70:	eef1 2946 	vneg.f16	s5, s12
  74:	eeb5 1940 	vcmp.f16	s2, #0.0
  78:	eeb5 19c0 	vcmpe.f16	s2, #0.0
  7c:	eef4 2966 	vcmp.f16	s5, s13
  80:	eef4 29e6 	vcmpe.f16	s5, s13
  84:	fe46 29ab 	vseleq.f16	s5, s13, s23
  88:	fe66 29ab 	vselge.f16	s5, s13, s23
  8c:	fe56 29ab 	vselvs.f16	s5, s13, s23
  90:	eefd 19c4 	vcvt.s32.f16	s3, s8
  94:	eefc 19c4 	vcvt.u32.f16	s3, s8
  98:	eef8 19c4 	vcvt.f16.s32	s3, s8
  9c:	eef8 1944 	vcvt.f16.u32	s3, s8
  a0:	eefa 39e1 	vcvt.f16.s32	s7, s7, #29
  a4:	eefb 39e1 	vcvt.f16.u32	s7, s7, #29
  a8:	eefe 39e1 	vcvt.s32.f16	s7, s7, #29
  ac:	eeff 39e1 	vcvt.u32.f16	s7, s7, #29
  b0:	fefc 29c5 	vcvta.s32.f16	s5, s10
  b4:	fefc 2945 	vcvta.u32.f16	s5, s10
  b8:	feff 29c5 	vcvtm.s32.f16	s5, s10
  bc:	feff 2945 	vcvtm.u32.f16	s5, s10
  c0:	fefd 29c5 	vcvtn.s32.f16	s5, s10
  c4:	fefd 2945 	vcvtn.u32.f16	s5, s10
  c8:	fefe 29c5 	vcvtp.s32.f16	s5, s10
  cc:	fefe 2945 	vcvtp.u32.f16	s5, s10
  d0:	eefc 2945 	vcvtr.u32.f16	s5, s10
  d4:	eefd 2945 	vcvtr.s32.f16	s5, s10
  d8:	fef8 1965 	vrinta.f16	s3, s11
  dc:	fefb 1965 	vrintm.f16	s3, s11
  e0:	fef9 1965 	vrintn.f16	s3, s11
  e4:	fefa 1965 	vrintp.f16	s3, s11
  e8:	eef6 1965 	vrintr.f16	s3, s11
  ec:	eef7 1965 	vrintx.f16	s3, s11
  f0:	eef6 19e5 	vrintz.f16	s3, s11
  f4:	fef0 2ae4 	vins.f16	s5, s9
  f8:	fef0 2a64 	vmovx.f16	s5, s9
  fc:	eeba 0948 	vcvt.f16.s16	s0, s0, #0
 100:	eebb 0948 	vcvt.f16.u16	s0, s0, #0
 104:	eebe 0948 	vcvt.s16.f16	s0, s0, #0
 108:	eebf 0948 	vcvt.u16.f16	s0, s0, #0
 10c:	eeba 0946 	vcvt.f16.s16	s0, s0, #4
 110:	eebb 0946 	vcvt.f16.u16	s0, s0, #4
 114:	eebe 0946 	vcvt.s16.f16	s0, s0, #4
 118:	eebf 0946 	vcvt.u16.f16	s0, s0, #4
 11c:	eeba 0940 	vcvt.f16.s16	s0, s0, #16
 120:	eebb 0940 	vcvt.f16.u16	s0, s0, #16
 124:	eebe 0940 	vcvt.s16.f16	s0, s0, #16
 128:	eebf 0940 	vcvt.u16.f16	s0, s0, #16
 12c:	eefa 2948 	vcvt.f16.s16	s5, s5, #0
 130:	eefb 2948 	vcvt.f16.u16	s5, s5, #0
 134:	eefe 2948 	vcvt.s16.f16	s5, s5, #0
 138:	eeff 2948 	vcvt.u16.f16	s5, s5, #0
 13c:	eeba 3946 	vcvt.f16.s16	s6, s6, #4
 140:	eebb 3946 	vcvt.f16.u16	s6, s6, #4
 144:	eebe 3946 	vcvt.s16.f16	s6, s6, #4
 148:	eebf 3946 	vcvt.u16.f16	s6, s6, #4
 14c:	eefa 3940 	vcvt.f16.s16	s7, s7, #16
 150:	eefb 3940 	vcvt.f16.u16	s7, s7, #16
 154:	eefe 3940 	vcvt.s16.f16	s7, s7, #16
 158:	eeff 3940 	vcvt.u16.f16	s7, s7, #16
 15c:	eefa 7948 	vcvt.f16.s16	s15, s15, #0
 160:	eefb 7948 	vcvt.f16.u16	s15, s15, #0
 164:	eefe 7948 	vcvt.s16.f16	s15, s15, #0
 168:	eeff 7948 	vcvt.u16.f16	s15, s15, #0
 16c:	eefa 7946 	vcvt.f16.s16	s15, s15, #4
 170:	eefb 7946 	vcvt.f16.u16	s15, s15, #4
 174:	eefe 7946 	vcvt.s16.f16	s15, s15, #4
 178:	eeff 7946 	vcvt.u16.f16	s15, s15, #4
 17c:	eefa 7940 	vcvt.f16.s16	s15, s15, #16
 180:	eefb 7940 	vcvt.f16.u16	s15, s15, #16
 184:	eefe 7940 	vcvt.s16.f16	s15, s15, #16
 188:	eeff 7940 	vcvt.u16.f16	s15, s15, #16
