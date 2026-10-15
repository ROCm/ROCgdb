#source: reloc-abs32-1.s
#source: reloc-abs32-2.s
#source: reloc-abs32-3.s
#ld: -T reloc-abs32.t
#objdump: -s -j .rodata
#name: 32-bit absolute relocations

# target is at 0x1002 and local at 0x1008.  The first two words and the
# last one use R_XTENSA_32_ABS and take their addends from r_addend; the
# third uses the partial_inplace R_XTENSA_32 and takes its addend of 8 from
# the relocated word.  Both byte orders are accepted.

.*: +file format .*xtensa.*

Contents of section \.rodata:
 100c (00001002 0000100a 0000100a 0000100c|02100000 0a100000 0a100000 0c100000).*
