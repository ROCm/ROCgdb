#source: table-jump.s
#as: -march=rv64ic_zcmt --defsym TEST_CM_JT_JALT=1
#ld: -Ttable-jump.ld --no-relax-tbljal
#objdump: -d
#...
Disassembly of section .text:
0+[0-9a-f]+ <_start>:
.*:[ 	]+[0-9a-f]+[ 	]+j[ 	]+[0-9a-f]+ <foo_global>
.*:[ 	]+[0-9a-f]+[ 	]+j[ 	]+[0-9a-f]+ <foo_global>
.*:[ 	]+[0-9a-f]+[ 	]+j[ 	]+[0-9a-f]+ <foo_global>
.*:[ 	]+[0-9a-f]+[ 	]+j[ 	]+[0-9a-f]+ <foo_global>
.*:[ 	]+[0-9a-f]+[ 	]+j[ 	]+[0-9a-f]+ <foo_global>
.*:[ 	]+[0-9a-f]+[ 	]+jal[ 	]+[0-9a-f]+ <foo_global>
.*:[ 	]+[0-9a-f]+[ 	]+jal[ 	]+[0-9a-f]+ <foo_global>
.*:[ 	]+[0-9a-f]+[ 	]+jal[ 	]+[0-9a-f]+ <foo_global>
.*:[ 	]+[0-9a-f]+[ 	]+jal[ 	]+[0-9a-f]+ <foo_global>
#...
