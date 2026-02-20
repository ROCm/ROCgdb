#objdump: -dr --prefix-addresses --show-raw-insn
#name: PRU mvi

# Test the mvi instruction

.*: +file format elf32-pru

Disassembly of section .text:
0+0000 <[^>]*> 2c204198 	mvib	r24.w0, \*r1.b2
0+0004 <[^>]*> 2ca20101 	mvid	\*r1.b0, \*r1.b0
0+0008 <[^>]*> 2ce20101 	mvid	\*r1.b0, \*--r1.b0
0+000c <[^>]*> 2dc20101 	mvid	\*--r1.b0, \*r1.b0\+\+
0+0010 <[^>]*> 2cc14101 	mviw	\*r1.b0, \*r1.b2\+\+
0+0014 <[^>]*> 2cc14121 	mviw	\*r1.b1, \*r1.b2\+\+
0+0018 <[^>]*> 2c4261f4 	mvid	r20, \*r1.b3\+\+
0+001c <[^>]*> 2c6241f4 	mvid	r20, \*--r1.b2
0+0020 <[^>]*> 2d82ff41 	mvid	\*--r1.b2, r31
