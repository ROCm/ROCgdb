#source: table-jump.s
#as: -march=rv64ic_zcmt --defsym TEST_CM_JT_JALT=1
#ld: --relax-tbljal
#objdump: -d

#...
Disassembly of section .text:
#...
.*:[ 	]+[0-9a-f]+[ 	]+cm.jt[ 	]+0 # [0-9a-f]+ <foo_global>
#...
.*:[ 	]+[0-9a-f]+[ 	]+cm.jalt[ 	]+32 # [0-9a-f]+ <foo_global>
#pass
