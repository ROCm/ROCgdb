#source: table-jump-gc-dead.s
#as: -march=rv64ic_zcmt
#ld: --gc-sections --relax-tbljal -e _start
#objdump: -d

#...
Disassembly of section .text:
#...
0+[0-9a-f]+ <_start>:
.*:[ 	]+[0-9a-f]+[ 	]+ret
#pass
