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
 *([\da-f]+):	ec 12 00 00 00 45 [	 ]*brxlg	%r1,%r2,\1 <foo\+0x\1>
.*: R_390_PC16DBL	test_rie\+0x2
 *([\da-f]+):	c0 e5 00 00 00 00 [	 ]*brasl	%r14,\1 <foo\+0x\1>
.*: R_390_PC32DBL	test_ril\+0x2
.*:	eb 12 40 03 00 0d [	 ]*sllg	%r1,%r2,3\(%r4\)
.*:	c8 50 20 01 40 03 [	 ]*mvcos	1\(%r2\),3\(%r4\),%r5
#?.*:	07 07 [	 ]*nopr	%r7
