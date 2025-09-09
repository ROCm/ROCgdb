#name: a "middle" memory is too small to fit anything, the sections are moved to the next available memory.
## Use case description:
## - sections .code.1, .code.2 and .code.3 (+ farcall stub) fit in .raml
## - section .code.4 fits in .ramz
## - nothing fits in .ramu
#source: non-contiguous-mem-1.s
#ld: --enable-non-contiguous-regions -T non-contiguous-ok-4.ld
#objdump: -rdth

.*:     file format elf64-(little|big)aarch64

Sections:
Idx Name          Size      VMA               LMA               File off  Algn
  0 \.raml         00000040  000000001fff0000  000000001fff0000  00010000  2\*\*3
                  CONTENTS, ALLOC, LOAD, READONLY, CODE
  1 \.ramz         00000050  0000000040040000  0000000030000000  00020000  2\*\*2
                  CONTENTS, ALLOC, LOAD, READONLY, CODE
SYMBOL TABLE:
000000001fff0000 l    d  \.raml	0000000000000000 \.raml
0000000040040000 l    d  \.ramz	0000000000000000 \.ramz
0000000000000000 l    df \*ABS\*	0000000000000000 non-contiguous-mem-.*\.o
000000001fff0028 l     F \.raml	000000000000000c __code4_veneer
000000001fff0040 g       \.raml	0000000000000000 _raml_end
0000000030000000 g       \.raml	0000000000000000 _ramu_start
000000001fff000c g     F \.raml	000000000000000c code2
0000000040040000 g       \.ramz	0000000000000000 _ramz_start
000000001fff0000 g       \.raml	0000000000000000 _raml_start
000000001fff0018 g     F \.raml	0000000000000008 code3
000000001fff0000 g     F \.raml	000000000000000c code1
0000000040040050 g       \.ramz	0000000000000000 _ramz_end
0000000040040000 g     F \.ramz	0000000000000050 code4
0000000030000000 g       \.raml	0000000000000000 _ramu_end



Disassembly of section \.raml:

000000001fff0000 \<code1\>:
    1fff0000:	d503201f 	nop
    1fff0004:	d503201f 	nop
    1fff0008:	94000001 	bl	1fff000c \<code2\>

000000001fff000c \<code2\>:
    1fff000c:	d503201f 	nop
    1fff0010:	d503201f 	nop
    1fff0014:	94000001 	bl	1fff0018 \<code3\>

000000001fff0018 \<code3\>:
    1fff0018:	d503201f 	nop
    1fff001c:	94000003 	bl	1fff0028 \<__code4_veneer\>
    1fff0020:	14000008 	b	1fff0040 \<_raml_end\>
    1fff0024:	d503201f 	nop

000000001fff0028 \<__code4_veneer\>:
    1fff0028:	90100290 	adrp	x16, 40040000 \<code4\>
    1fff002c:	91000210 	add	x16, x16, #0x0
    1fff0030:	d61f0200 	br	x16
	...

Disassembly of section \.ramz:

0000000040040000 \<code4\>:
    40040000:	d503201f 	nop
    40040004:	d503201f 	nop
    40040008:	d503201f 	nop
    4004000c:	d503201f 	nop
    40040010:	d503201f 	nop
    40040014:	d503201f 	nop
    40040018:	d503201f 	nop
    4004001c:	d503201f 	nop
    40040020:	d503201f 	nop
    40040024:	d503201f 	nop
    40040028:	d503201f 	nop
    4004002c:	d503201f 	nop
    40040030:	d503201f 	nop
    40040034:	d503201f 	nop
    40040038:	d503201f 	nop
    4004003c:	d503201f 	nop
    40040040:	d503201f 	nop
    40040044:	d503201f 	nop
    40040048:	d503201f 	nop
    4004004c:	d503201f 	nop
