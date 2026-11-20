#source: table-jump-shared.s
#as: -march=rv64ic_zcmt -fpic
#ld: -shared --relax-tbljal
#objdump: -d
#...
Disassembly of section .text:
#...
0+[0-9a-f]+ <foo>:
#  Calls should NOT be cm.jalt in shared library (should be jal to PLT)
.*:[ 	]+[0-9a-f]+[ 	]+jal[ 	]+[0-9a-f]+ <bar@plt>
.*:[ 	]+[0-9a-f]+[ 	]+jal[ 	]+[0-9a-f]+ <bar@plt>
#pass
