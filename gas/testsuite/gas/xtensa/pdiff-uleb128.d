#objdump: -r -s -j .debug_info
#name: uleb128 difference reloc

.*: +file format .*xtensa.*

RELOCATION RECORDS FOR \[.debug_info\]:
OFFSET +TYPE +VALUE
0+ R_XTENSA_PDIFF_ULEB128 +.text.*

# Difference is 200 (0xc8 0x01) plus one spare byte reserved for growth.
Contents of section .debug_info:
 0000 c88100.*
