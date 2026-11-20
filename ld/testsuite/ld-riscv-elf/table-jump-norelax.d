#source: table-jump-norelax.s
#as: -march=rv64ic_zcmt
#ld: -Ttable-jump.ld --relax-tbljal
#objdump: -d
#...
Disassembly of section .text:
0+[0-9a-f]+ <_start>:
.*:[ 	]+[0-9a-f]+[ 	]+auipc[ 	]+ra,0x1
.*:[ 	]+[0-9a-f]+[ 	]+jalr[ 	]+[0-9]+\(ra\) # [0-9a-f]+ <target>
.*:[ 	]+[0-9a-f]+[ 	]+auipc[ 	]+t[0-9],0x1
.*:[ 	]+[0-9a-f]+[ 	]+jr[ 	]+[0-9]+\(t[0-9]\) # [0-9a-f]+ <target>
.*:[ 	]+[0-9a-f]+[ 	]+j[ 	]+[0-9a-f]+ <target>
#...
0+[0-9a-f]+ <target>:
.*:[ 	]+[0-9a-f]+[ 	]+ret
