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
.*:	eb 12 40 03 00 0d [	 ]*sllg	%r1,%r2,3\(%r4\)
.*:	c8 50 20 01 40 03 [	 ]*mvcos	1\(%r2\),3\(%r4\),%r5
#?.*:	07 07 [	 ]*nopr	%r7
