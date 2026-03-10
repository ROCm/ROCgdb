# name: all sections fit in the available memories + no farcall stub.
## Use case description:
## - sections .code.1 and .code.2 fit in .raml
## - section .code.3 fits in .ramu and does not need a farcall stub to jump
##   to code4
## - section .code.4 fits in .ramz
# source: non-contiguous-mem-1.s
# ld: --enable-non-contiguous-regions -T non-contiguous-ok-1.ld
# objdump: -rdth

.*:     file format elf64-(little|big)aarch64

Sections:
Idx Name          Size      VMA               LMA               File off  Algn
  0 \.raml         00000018  000000001fff0000  000000001fff0000  00010000  2\*\*2
                  CONTENTS, ALLOC, LOAD, READONLY, CODE
  1 \.ramu         00000008  0000000020000000  000000001fff0018  00020000  2\*\*2
                  CONTENTS, ALLOC, LOAD, READONLY, CODE
  2 \.ramz         00000050  0000000020040000  0000000020000008  00030000  2\*\*2
                  CONTENTS, ALLOC, LOAD, READONLY, CODE
SYMBOL TABLE:
000000001fff0000 l    d  \.raml	0000000000000000 \.raml
0000000020000000 l    d  \.ramu	0000000000000000 \.ramu
0000000020040000 l    d  \.ramz	0000000000000000 \.ramz
0000000000000000 l    df \*ABS\*	0000000000000000 non-contiguous-mem-.*\.o
000000001fff0018 g       \.raml	0000000000000000 _raml_end
0000000020000000 g       \.ramu	0000000000000000 _ramu_start
000000001fff000c g     F \.raml	000000000000000c code2
0000000020040000 g       \.ramz	0000000000000000 _ramz_start
000000001fff0000 g       \.raml	0000000000000000 _raml_start
0000000020000000 g     F \.ramu	0000000000000008 code3
000000001fff0000 g     F \.raml	000000000000000c code1
0000000020040050 g       \.ramz	0000000000000000 _ramz_end
0000000020040000 g     F \.ramz	0000000000000050 code4
0000000020000008 g       \.ramu	0000000000000000 _ramu_end



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
    20000004:	9400ffff 	bl	20040000 \<code4\>

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
