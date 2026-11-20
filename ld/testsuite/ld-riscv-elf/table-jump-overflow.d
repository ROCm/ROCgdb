#source: table-jump-overflow.s
#as: -march=rv64ic_zcmt
#ld: -Ttable-jump.ld --relax-tbljal
#objdump: -d
#...
Disassembly of section .text:
0+[0-9a-f]+ <_start>:
.*:[ 	]+[0-9a-f]+[ 	]+jal[ 	]+[0-9a-f]+ <func_top>
#...
.*:[ 	]+[0-9a-f]+[ 	]+jal[ 	]+[0-9a-f]+ <func_overflow>
#pass
