#name: subsequent sections should not be swapped, even if they fit in a previous hole.
## Use case description:
## - section .code.1 fits in .raml
## - section .code.2 does not fit in .raml and goes to .ramu
## - section .code.3 would fit in .raml, but goes to .ramu:  Check that .code.2 and .code.3 are not swapped
## - section .code.4 fits in .ramz
#source: non-contiguous-mem-1.s
#ld: --enable-non-contiguous-regions -T non-contiguous-ok-3.ld
#objdump: -rdth

.*:     file format elf64-(little|big)aarch64

Sections:
Idx Name          Size      VMA               LMA               File off  Algn
  0 \.raml         0000000c  000000001fff0000  000000001fff0000  00010000  2\*\*2
                  CONTENTS, ALLOC, LOAD, READONLY, CODE
  1 \.ramu         00000014  0000000020000000  000000001fff000c  00020000  2\*\*2
                  CONTENTS, ALLOC, LOAD, READONLY, CODE
  2 \.ramz         00000050  0000000020040000  0000000020000014  00030000  2\*\*2
                  CONTENTS, ALLOC, LOAD, READONLY, CODE
SYMBOL TABLE:
000000001fff0000 l    d  \.raml	0000000000000000 \.raml
0000000020000000 l    d  \.ramu	0000000000000000 \.ramu
0000000020040000 l    d  \.ramz	0000000000000000 \.ramz
0000000000000000 l    df \*ABS\*	0000000000000000 non-contiguous-mem-.*\.o
000000001fff000c g       \.raml	0000000000000000 _raml_end
0000000020000000 g       \.ramu	0000000000000000 _ramu_start
0000000020000000 g     F \.ramu	000000000000000c code2
0000000020040000 g       \.ramz	0000000000000000 _ramz_start
000000001fff0000 g       \.raml	0000000000000000 _raml_start
000000002000000c g     F \.ramu	0000000000000008 code3
000000001fff0000 g     F \.raml	000000000000000c code1
0000000020040050 g       \.ramz	0000000000000000 _ramz_end
0000000020040000 g     F \.ramz	0000000000000050 code4
0000000020000014 g       \.ramu	0000000000000000 _ramu_end



Disassembly of section \.raml:

000000001fff0000 \<code1\>:
    1fff0000:	d503201f 	nop
    1fff0004:	d503201f 	nop
    1fff0008:	94003ffe 	bl	20000000 \<code2\>

Disassembly of section \.ramu:

0000000020000000 \<code2\>:
    20000000:	d503201f 	nop
    20000004:	d503201f 	nop
    20000008:	94000001 	bl	2000000c \<code3\>

000000002000000c \<code3\>:
    2000000c:	d503201f 	nop
    20000010:	9400fffc 	bl	20040000 \<code4\>

Disassembly of section \.ramz:

0000000020040000 \<code4\>:
    20040000:	d503201f 	nop
    20040004:	d503201f 	nop
    20040008:	d503201f 	nop
    2004000c:	d503201f 	nop
    20040010:	d503201f 	nop
    20040014:	d503201f 	nop
    20040018:	d503201f 	nop
    2004001c:	d503201f 	nop
    20040020:	d503201f 	nop
    20040024:	d503201f 	nop
    20040028:	d503201f 	nop
    2004002c:	d503201f 	nop
    20040030:	d503201f 	nop
    20040034:	d503201f 	nop
    20040038:	d503201f 	nop
    2004003c:	d503201f 	nop
    20040040:	d503201f 	nop
    20040044:	d503201f 	nop
    20040048:	d503201f 	nop
    2004004c:	d503201f 	nop
