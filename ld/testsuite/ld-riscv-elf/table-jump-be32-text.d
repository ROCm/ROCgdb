#source: table-jump-rv32.s
#as: -march=rv32ic_zcmt -mbig-endian
#ld: -melf32briscv -Ttable-jump.ld --relax-tbljal
#objdump: -d

#...
.*:     file format elf32-bigriscv
#...
Disassembly of section .text:
#...
.*:[ 	]+[0-9a-f]+[ 	]+cm.jt[ 	]+0 # [0-9a-f]+ <foo_local>
.*:[ 	]+[0-9a-f]+[ 	]+cm.jt[ 	]+1 # [0-9a-f]+ <foo_global>
#pass
