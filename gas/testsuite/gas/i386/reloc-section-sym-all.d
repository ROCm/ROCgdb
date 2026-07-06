#source: reloc-section-sym.s
#as: --reloc-section-sym=all
#objdump: -rsj .data -j .text1
#name: reloc-section-sym=all

.*:     file format .*

RELOCATION RECORDS FOR \[\.data\]:
OFFSET +TYPE +VALUE
0+0 R_X86_64_64 +\.text\+0x0+11
0+8 R_X86_64_64 +\.text\+0x0+12


RELOCATION RECORDS FOR \[\.text1\]:
OFFSET +TYPE +VALUE
0+1 R_X86_64_PC32 +\.text-0x0+3
0+6 R_X86_64_PC32 +\.text-0x0+2


Contents of section \.data:
 0000 00000000 00000000 00000000 00000000  \.+
Contents of section \.text1:
 0000 e8000000 00e80000 0000 +.*
