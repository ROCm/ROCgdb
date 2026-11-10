#name: s390 operands
#objdump: -dr

.*: +file format .*

Disassembly of section .text:

.* <foo>:
# E
.*:	01 01 [	 ]*pr
# I
.*:	0a 01 [	 ]*svc	1
# RI
.*:	a7 1a 80 01 [	 ]*ahi	%r1,-32767
# RR
.*:	18 12 [	 ]*lr	%r1,%r2
# RRE
.*:	b2 5e 00 12 [	 ]*srst	%r1,%r2
# RRF
.*:	b3 5b 34 12 [	 ]*didbr	%f1,%f3,%f2,4
# RS
.*:	ba 12 40 03 [	 ]*cs	%r1,%r2,3\(%r4\)
# RSI
 *([\da-f]+):	84 12 00 00 [	 ]*brxh	%r1,%r2,\1 <foo\+0x\1>
.*: R_390_PC16DBL	test_rsi\+0x2
# RX
.*:	58 13 40 02 [	 ]*l	%r1,2\(%r3,%r4\)
# RXE
.*:	ed 10 30 02 00 1a [	 ]*adb	%f1,2\(%r3\)
# RXF
.*:	ed 24 50 03 10 1e [	 ]*madb	%f1,%f2,3\(%r4,%r5\)
# S
.*:	b2 33 20 01 [	 ]*ssch	1\(%r2\)
.*:	92 03 20 01 [	 ]*mvi	1\(%r2\),3
.*:	d2 26 30 01 50 04 [	 ]*mvc	1\(39,%r3\),4\(%r5\)
.*:	e5 01 20 01 40 03 [	 ]*tprot	1\(%r2\),3\(%r4\)
#?.*:	07 07 [	 ]*nopr	%r7
