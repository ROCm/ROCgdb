#name: all code sections fit in the available memories AND farcall stub to jump to code4 ALSO fits in .ramu
## Use case description:
## - sections .code.1 and .code.2 fit in .raml
## - both section .code.3 and farcall stub to jump to code4 fits in .ramu
## - section .code.4 fits in .ramz
#source: non-contiguous-mem-1.s
#ld: --enable-non-contiguous-regions -T non-contiguous-ok-2.ld
#objdump: -rdth

.*:     file format elf64-(little|big)aarch64

Sections:
Idx Name          Size      VMA               LMA               File off  Algn
  0 \.raml         00000018  000000001fff0000  000000001fff0000  00010000  2\*\*2
                  CONTENTS, ALLOC, LOAD, READONLY, CODE
  1 \.ramu         00000028  0000000020000000  000000001fff0018  00020000  2\*\*3
                  CONTENTS, ALLOC, LOAD, READONLY, CODE
  2 \.ramz         00000050  0000000030040000  0000000020000028  00030000  2\*\*2
                  CONTENTS, ALLOC, LOAD, READONLY, CODE
SYMBOL TABLE:
000000001fff0000 l    d  \.raml	0000000000000000 \.raml
0000000020000000 l    d  \.ramu	0000000000000000 \.ramu
0000000030040000 l    d  \.ramz	0000000000000000 \.ramz
0000000000000000 l    df \*ABS\*	0000000000000000 non-contiguous-mem-.*\.o
0000000020000010 l     F \.ramu	000000000000000c __code4_veneer
000000001fff0018 g       \.raml	0000000000000000 _raml_end
0000000020000000 g       \.ramu	0000000000000000 _ramu_start
000000001fff000c g     F \.raml	000000000000000c code2
0000000030040000 g       \.ramz	0000000000000000 _ramz_start
000000001fff0000 g       \.raml	0000000000000000 _raml_start
0000000020000000 g     F \.ramu	0000000000000008 code3
000000001fff0000 g     F \.raml	000000000000000c code1
0000000030040050 g       \.ramz	0000000000000000 _ramz_end
0000000030040000 g     F \.ramz	0000000000000050 code4
0000000020000028 g       \.ramu	0000000000000000 _ramu_end



Disassembly of section \.raml:

000000001fff0000 \<code1\>:
    1fff0000:	d503201f 	nop
    1fff0004:	d503201f 	nop
    1fff0008:	94000001 	bl	1fff000c \<code2\>

000000001fff000c \<code2\>:
    1fff000c:	d503201f 	nop
    1fff0010:	d503201f 	nop
    1fff0014:	94003ffb 	bl	20000000 \<code3\>

Disassembly of section \.ramu:

0000000020000000 \<code3\>:
    20000000:	d503201f 	nop
    20000004:	94000003 	bl	20000010 \<__code4_veneer\>
    20000008:	14000008 	b	20000028 \<_ramu_end\>
    2000000c:	d503201f 	nop

0000000020000010 \<__code4_veneer\>:
    20000010:	90080210 	adrp	x16, 30040000 \<code4\>
    20000014:	91000210 	add	x16, x16, #0x0
    20000018:	d61f0200 	br	x16
	\.\.\.

Disassembly of section \.ramz:

0000000030040000 \<code4\>:
    30040000:	d503201f 	nop
    30040004:	d503201f 	nop
    30040008:	d503201f 	nop
    3004000c:	d503201f 	nop
    30040010:	d503201f 	nop
    30040014:	d503201f 	nop
    30040018:	d503201f 	nop
    3004001c:	d503201f 	nop
    30040020:	d503201f 	nop
    30040024:	d503201f 	nop
    30040028:	d503201f 	nop
    3004002c:	d503201f 	nop
    30040030:	d503201f 	nop
    30040034:	d503201f 	nop
    30040038:	d503201f 	nop
    3004003c:	d503201f 	nop
    30040040:	d503201f 	nop
    30040044:	d503201f 	nop
    30040048:	d503201f 	nop
    3004004c:	d503201f 	nop
