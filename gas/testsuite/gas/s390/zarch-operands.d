#name: s390x operands
#objdump: -dr

.*: +file format .*

Disassembly of section .text:

.* <foo>:
# E
.*:	01 01 [	 ]*pr
# I
.*:	0a 01 [	 ]*svc	1
# IE
.*:	b2 fa 00 12 [	 ]*niai	1,2
# MII
 *([\da-f]+):	c5 10 00 00 00 00 [	 ]*bprp	1,\1 <foo\+0x\1>,\1 <foo\+0x\1>
.*: R_390_PC12DBL	test_mii\+0x1
.*: R_390_PC24DBL	test_mii\+0x3
# RI
.*:	a7 1a 00 01 [	 ]*ahi	%r1,1
# RI-a
.*:	a7 1a ff ff [	 ]*ahi	%r1,-1
# RI-b
 *([\da-f]+):	a7 16 00 00 [	 ]*brct	%r1,\1 <foo\+0x\1>
.*: R_390_PC16DBL	test_ri_b\+0x2
# RI-c
 *([\da-f]+):	a7 f4 00 00 [	 ]*j	\1 <foo\+0x\1>
.*: R_390_PC16DBL	test_ri_c\+0x2
# RIE
 *([\da-f]+):	ec 12 00 00 00 45 [	 ]*brxlg	%r1,%r2,\1 <foo\+0x\1>
.*: R_390_PC16DBL	test_rie\+0x2
# RIE-a
.*:	ec 10 ff ff 70 72 [	 ]*cit	%r1,-1,7
# RIE-b
 *([\da-f]+):	ec 12 00 00 70 76 [	 ]*crj	%r1,%r2,7,\1 <foo\+0x\1>
# RIE-c
 *([\da-f]+):	ec 17 00 00 ff 7e [	 ]*cij	%r1,-1,7,\1 <foo\+0x\1>
# RIE-d
.*:	ec 13 ff ff 00 d8 [	 ]*ahik	%r1,%r3,-1
# RIE-e
 *([\da-f]+):	ec 13 00 00 00 44 [	 ]*brxhg	%r1,%r3,\1 <foo\+0x\1>
# RIE-f
.*:	ec 12 03 04 05 54 [	 ]*rnsbg	%r1,%r2,3,4,5
# RIE-g
.*:	ec 1f ff ff 00 42 [	 ]*lochi	%r1,-1,15
# RIL
 *([\da-f]+):	c0 e5 00 00 00 00 [	 ]*brasl	%r14,\1 <foo\+0x\1>
.*: R_390_PC32DBL	test_ril\+0x2
.*:	c2 19 00 00 00 01 [	 ]*afi	%r1,1
# RIL-a
.*:	c2 19 ff ff ff ff [	 ]*afi	%r1,-1
# RIL-b
 *([\da-f]+):	c0 e5 00 00 00 00 [	 ]*brasl	%r14,\1 <foo\+0x\1>
# RIL-c
 *([\da-f]+):	c0 f4 00 00 00 00 [	 ]*jg	\1 <foo\+0x\1>
# RIS
.*:	ec 13 41 bc fe fe [	 ]*cib	%r1,-2,3,444\(%r4\)
# RR
.*:	1a 12 [	 ]*ar	%r1,%r2
.*:	04 10 [	 ]*spm	%r1
# RRD
.*:	b3 2e 10 32 [	 ]*maer	%f1,%f3,%f2
# RRE
.*:	b9 08 00 12 [	 ]*agr	%r1,%r2
.*:	b2 22 00 10 [	 ]*ipm	%r1
.*:	b2 48 00 00 [	 ]*palb
# RRF
.*:	b3 5b 34 12 [	 ]*didbr	%f1,%f3,%f2,4
# RRF-a
.*:	b2 21 00 12 [	 ]*ipte	%r1,%r2
.*:	b2 21 30 12 [	 ]*ipte	%r1,%r2,%r3
.*:	b2 21 34 12 [	 ]*ipte	%r1,%r2,%r3,4
# RRF-b
.*:	b9 8e 30 12 [	 ]*idte	%r1,%r3,%r2
.*:	b9 8e 34 12 [	 ]*idte	%r1,%r3,%r2,4
# RRF-c
.*:	b9 e1 00 12 [	 ]*popcnt	%r1,%r2
.*:	b9 e1 30 12 [	 ]*popcnt	%r1,%r2,3
# RRF-d
.*:	b3 dc 04 12 [	 ]*lxdtr	%f1,%f2,4
# RRF-e
.*:	b3 ca 30 12 [	 ]*cgxr	%r1,3,%f2
.*:	b3 a6 34 12 [	 ]*cxgbra	%f1,3,%r2,4
# RRS
.*:	ec 12 41 bc 30 f6 [	 ]*crb	%r1,%r2,3,444\(%r4\)
# RS
.*:	ba 12 31 4d [	 ]*cs	%r1,%r2,333\(%r3\)
# RS-a
.*:	86 13 20 de [	 ]*bxh	%r1,%r3,222\(%r2\)
.*:	8f 10 20 de [	 ]*slda	%r1,222\(%r2\)
# RS-b
.*:	bd 13 20 de [	 ]*clm	%r1,3,222\(%r2\)
# RSE
.*:	eb 13 20 de 00 0d [	 ]*sllg	%r1,%r3,222\(%r2\)
# RSI
 *([\da-f]+):	84 13 00 00 [	 ]*brxh	%r1,%r3,\1 <foo\+0x\1>
# RSL-a
.*:	eb a0 10 6f 00 c0 [	 ]*tp	111\(11,%r1\)
# RSL-b
.*:	ed 15 20 de 13 ae [	 ]*cdpt	%f1,222\(22,%r2\),3
# RSY
.*:	eb 13 26 ce 05 1d [	 ]*rll	%r1,%r3,22222\(%r2\)
# RSY-a
.*:	eb 13 2f ff ff 1d [	 ]*rll	%r1,%r3,-1\(%r2\)
# RSY-b
.*:	eb 13 2f ff ff 81 [	 ]*icmy	%r1,3,-1\(%r2\)
# RX
.*:	58 13 20 de [	 ]*l	%r1,222\(%r3,%r2\)
# RX-a
.*:	5a 13 20 de [	 ]*a	%r1,222\(%r3,%r2\)
# RX-b
.*:	47 13 20 de [	 ]*bo	222\(%r3,%r2\)
# RXE
.*:	ed 13 20 de 00 25 [	 ]*lxd	%f1,222\(%r3,%r2\)
.*:	e7 13 20 de 30 27 [	 ]*lcbb	%r1,222\(%r3,%r2\),3
# RXF
.*:	ed 34 20 de 10 3e [	 ]*mad	%f1,%f3,222\(%r4,%r2\)
# RXY
.*:	e3 12 36 ce 05 08 [	 ]*ag	%r1,22222\(%r2,%r3\)
# RXY-a
.*:	e3 12 36 ce 05 08 [	 ]*ag	%r1,22222\(%r2,%r3\)
# RXY-b
.*:	e3 12 36 ce 05 36 [	 ]*pfd	1,22222\(%r2,%r3\)
# RXY-c
.*:	e3 12 36 ce 05 60 [	 ]*lxab	%r1,22222\(%r2,%r3\)
# S
.*:	b2 33 10 6f [	 ]*ssch	111\(%r1\)
.*:	b2 31 00 00 [	 ]*hsch
# SI
.*:	95 16 10 6f [	 ]*cli	111\(%r1\),22
.*:	93 00 10 6f [	 ]*ts	111\(%r1\)
# SIL
.*:	e5 44 10 6f 56 ce [	 ]*mvhhi	111\(%r1\),22222
# SIY
.*:	eb 16 1b 67 02 7a [	 ]*agsi	11111\(%r1\),22
.*:	eb 00 1b 67 02 71 [	 ]*lpswey	11111\(%r1\)
# SMI
 *([\da-f]+):	c7 10 31 4d 00 00 [	 ]*bpp	1,\1 <foo\+0x\1>,333\(%r3\)
# SS
.*:	d9 14 20 6f 30 de [	 ]*mvck	111\(%r1,%r2\),222\(%r3\),%r4
# SS-a
.*:	d5 0a 10 6f 20 de [	 ]*clc	111\(11,%r1\),222\(%r2\)
# SS-b
.*:	f1 01 10 6f 20 de [	 ]*mvo	111\(1,%r1\),222\(2,%r2\)
# SS-c
.*:	f0 03 10 6f 20 de [	 ]*srp	111\(1,%r1\),222\(%r2\),3
# SS-d
.*:	d9 14 20 6f 30 de [	 ]*mvck	111\(%r1,%r2\),222\(%r3\),%r4
# SS-e
.*:	ef 13 20 de 41 bc [	 ]*lmd	%r1,%r3,222\(%r2\),444\(%r4\)
# SS-f
.*:	e1 15 10 6f 20 de [	 ]*pku	111\(%r1\),222\(22,%r2\)
# SSE
.*:	e5 0a 10 6f 20 de [	 ]*mvcrl	111\(%r1\),222\(%r2\)
# SSF
.*:	c8 30 10 6f 20 de [	 ]*mvcos	111\(%r1\),222\(%r2\),%r3
# VRI-a
.*:	e7 10 56 ce 00 44 [	 ]*vgbm	%v1,22222
.*:	e7 10 56 ce 30 40 [	 ]*vleib	%v1,22222,3
# VRI-b
.*:	e7 10 16 21 40 46 [	 ]*vgm	%v1,22,33,4
# VRI-c
.*:	e7 13 56 ce 40 4d [	 ]*vrep	%v1,%v3,22222,4
# VRI-d
.*:	e7 12 30 2c 00 86 [	 ]*vsld	%v1,%v2,%v3,44
.*:	e7 12 30 2c 50 72 [	 ]*verim	%v1,%v2,%v3,44,5
# VRI-e
.*:	e7 12 d0 55 40 4a [	 ]*vftci	%v1,%v2,3333,4,5
# VRI-f
.*:	e6 12 30 52 c0 71 [	 ]*vap	%v1,%v2,%v3,44,5
# VRI-g
.*:	e6 12 2c 52 10 5b [	 ]*vpsop	%v1,%v2,33,44,5
# VRI-h
.*:	e6 10 56 ce 30 49 [	 ]*vlip	%v1,22222,3
# VRI-i
.*:	e6 12 00 42 10 5a [	 ]*vcvdg	%v1,%r2,33,4
# VRI-j
.*:	e6 12 00 42 10 4a [	 ]*vcvdq	%v1,%v2,33,4
# VRI-k
.*:	e7 12 30 37 40 88 [	 ]*veval	%v1,%v2,%v3,%v4,55
# VRI-l
.*:	e6 01 28 23 50 7f [	 ]*vtz	%v1,%v2,33333
#?.*:	07 07 [	 ]*nopr	%r7
