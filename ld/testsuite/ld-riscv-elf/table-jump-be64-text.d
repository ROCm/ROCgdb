#source: table-jump.s
#as: -march=rv64ic_zcmt --defsym TEST_CM_JT=1 -mbig-endian
#ld: -melf64briscv -Ttable-jump.ld --relax-tbljal
#objdump: -d

#...
.*:     file format elf64-bigriscv
#...
Disassembly of section .text:
#...
.*:[ 	]+[0-9a-f]+[ 	]+cm.jt[ 	]+0 # [0-9a-f]+ <foo_local>
.*:[ 	]+[0-9a-f]+[ 	]+cm.jt[ 	]+1 # [0-9a-f]+ <foo_global>
#pass
