#source: reloc-section-sym.s
#as: --reloc-section-sym=none
#objdump: -rsj .data -j .text1
#name: reloc-section-sym=none

.*:     file format .*

RELOCATION RECORDS FOR \[\.data\]:
OFFSET +TYPE +VALUE
0+0 R_X86_64_64 +named_local\+0x0+10
0+8 R_X86_64_64 +\.Ltemp\+0x0+10


RELOCATION RECORDS FOR \[\.text1\]:
OFFSET +TYPE +VALUE
0+1 R_X86_64_PC32 +named_local-0x0+4
0+6 R_X86_64_PC32 +\.Ltemp-0x0+4


Contents of section \.data:
 0000 00000000 00000000 00000000 00000000  \.+
Contents of section \.text1:
 0000 e8000000 00e80000 0000 +.*
