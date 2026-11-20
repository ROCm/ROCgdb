#source: table-jump.s
#as: -march=rv64ic_zcmt --defsym TEST_CM_JALT=1
#ld: -Ttable-jump.ld --relax-tbljal
#objdump: -d
#...
Disassembly of section .riscv.jvt:
0+[0-9a-f]+ <__jvt_base\$>:
[ 	]+...
.*:[ 	]+[0-9a-f]+[ 	]+jvt.jalt\[32\]:[ 	]+foo_global
.*:[ 	]+[0-9a-f]+[ 	]+jvt.jalt\[33\]:[ 	]+foo_local
#...
Disassembly of section .text:
0+[0-9a-f]+ <_start>:
.*:[ 	]+[0-9a-f]+[ 	]+cm.jalt[ 	]+32 # [0-9a-f]+ <foo_global>
.*:[ 	]+[0-9a-f]+[ 	]+cm.jalt[ 	]+33 # [0-9a-f]+ <foo_local>
.*:[ 	]+[0-9a-f]+[ 	]+cm.jalt[ 	]+32 # [0-9a-f]+ <foo_global>
.*:[ 	]+[0-9a-f]+[ 	]+cm.jalt[ 	]+33 # [0-9a-f]+ <foo_local>
#...
