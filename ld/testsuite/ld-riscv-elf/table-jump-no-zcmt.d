#source: table-jump-no-zcmt.s
#as: -march=rv64ic
#ld: -Ttable-jump.ld --relax-tbljal
#objdump: -d
#...
Disassembly of section .text:
0+[0-9a-f]+ <_start>:
# All calls should remain as jal (not converted to cm.jalt)
.*:[ 	]+[0-9a-f]+[ 	]+jal[ 	]+[0-9a-f]+ <foo>
.*:[ 	]+[0-9a-f]+[ 	]+jal[ 	]+[0-9a-f]+ <foo>
.*:[ 	]+[0-9a-f]+[ 	]+jal[ 	]+[0-9a-f]+ <foo>
.*:[ 	]+[0-9a-f]+[ 	]+jal[ 	]+[0-9a-f]+ <foo>
#...
0+[0-9a-f]+ <foo>:
.*:[ 	]+[0-9a-f]+[ 	]+ret
#pass
