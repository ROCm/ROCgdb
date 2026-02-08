#source: reloc-section-sym.s
#as: --reloc-section-sym=none
#objdump: -rsj .data
#name: reloc-section-sym=none

.*: +file format .*

RELOCATION RECORDS FOR \[\.data\]:
OFFSET +TYPE +VALUE
0*0 [^ ]+ +local.*
0*4 [^ ]+ +\.Ltemp.*
#pass
