#as: --no-abs32-rela
#source: reloc-abs32.s
#objdump: -r -s -j .rodata
#name: 32-bit absolute relocations with --no-abs32-rela

# R_XTENSA_32 is partial_inplace, so the addends of references to the
# section symbol are held in the relocated words.  Both byte orders are
# accepted.

.*: +file format .*xtensa.*

RELOCATION RECORDS FOR \[\.rodata\]:
OFFSET +TYPE +VALUE
0+00 R_XTENSA_32 +\.text
0+04 R_XTENSA_32 +\.text
0+08 R_XTENSA_32 +gsym
0+0c R_XTENSA_32 +gsym\+0x0+8

Contents of section \.rodata:
 0000 (00000002 00000012|02000000 12000000) 00000000 00000000 .*
