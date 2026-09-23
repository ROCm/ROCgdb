#as: --abs32-rela
#objdump: -r -s -j .rodata
#name: 32-bit absolute relocations with --abs32-rela

# R_XTENSA_32_ABS keeps the addend in r_addend and leaves the relocated word
# zero.

.*: +file format .*xtensa.*

RELOCATION RECORDS FOR \[\.rodata\]:
OFFSET +TYPE +VALUE
0+00 R_XTENSA_32_ABS +\.text\+0x0+2
0+04 R_XTENSA_32_ABS +\.text\+0x0+12
0+08 R_XTENSA_32_ABS +gsym
0+0c R_XTENSA_32_ABS +gsym\+0x0+8

Contents of section \.rodata:
 0000 00000000 00000000 00000000 00000000 .*
