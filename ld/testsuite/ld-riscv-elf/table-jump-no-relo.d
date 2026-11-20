#source: table-jump-no-relo-1.s
#source: table-jump-no-relo-2.s
#as: -march=rv64ic_zcmt
#ld: -Ttable-jump.ld --relax-tbljal
#objdump: -d
#...
Disassembly of section .text:
0+[0-9a-f]+ <_start>:
.*:[ 	]+[0-9a-f]+[ 	]+j[ 	]+[0-9a-f]+ <_start>
.*:[ 	]+[0-9a-f]+[ 	]+ret

0+[0-9a-f]+ <foo>:
.*:[ 	]+[0-9a-f]+[ 	]+jal[ 	]+[0-9a-f]+ <bar>

0+[0-9a-f]+ <bar>:
.*:[ 	]+[0-9a-f]+[ 	]+ret
