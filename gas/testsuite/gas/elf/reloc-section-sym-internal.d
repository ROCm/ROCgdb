#source: reloc-section-sym.s
#as: --reloc-section-sym=internal
#objdump: -rsj .data
#name: reloc-section-sym=internal

.*: +file format .*

RELOCATION RECORDS FOR \[\.data\]:
OFFSET +TYPE +VALUE
0*0 [^ ]+ +local.*
0*4 [^ ]+ +\.bss.*
#pass
