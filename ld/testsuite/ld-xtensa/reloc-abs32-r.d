#source: reloc-abs32-1.s
#source: reloc-abs32-2.s
#source: reloc-abs32-3.s
#ld: -r
#objdump: -r -s -j .rodata
#name: 32-bit absolute relocations, relocatable link

# R_XTENSA_32_ABS keeps its addend in r_addend and leaves the relocated word
# zero, including the reference to .text whose addend is rebased by the 4
# bytes of .text that precede it.  The partial_inplace R_XTENSA_32 keeps its
# addend of 8 in the relocated word.  Both byte orders are accepted.

.*: +file format .*xtensa.*

RELOCATION RECORDS FOR \[\.rodata\]:
OFFSET +TYPE +VALUE
0+00 R_XTENSA_32_ABS +target
0+04 R_XTENSA_32_ABS +target\+0x0+8
0+08 R_XTENSA_32 +target
0+0c R_XTENSA_32_ABS +\.text\+0x0+c

Contents of section \.rodata:
 0000 00000000 00000000 (00000008|08000000) 00000000 .*
