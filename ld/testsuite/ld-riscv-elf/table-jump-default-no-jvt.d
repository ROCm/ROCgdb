#source: table-jump-no-benefit.s
#as: -march=rv64ic_zcmt
#ld: --relax-tbljal
#objdump: -d

#...
Disassembly of section .text:
#...
0+[0-9a-f]+ <_start>:
.*:[ 	]+[0-9a-f]+[ 	]+j[ 	]+[0-9a-f]+ <target>
.*:[ 	]+[0-9a-f]+[ 	]+j[ 	]+[0-9a-f]+ <target>
#...
0+[0-9a-f]+ <target>:
.*:[ 	]+[0-9a-f]+[ 	]+ret
#pass
