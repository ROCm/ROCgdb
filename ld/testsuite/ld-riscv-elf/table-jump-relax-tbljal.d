#source: table-jump.s
#as: -march=rv64ic_zcmt --defsym TEST_CM_JT_JALT=1
#ld: -Ttable-jump.ld --relax-tbljal
#objdump: -dz
#...
Disassembly of section .riscv.jvt:
0+[0-9a-f]+ <__jvt_base\$>:
.*:[ 	]+[0-9a-f]+[ 	]+jvt.jt\[0\]:[ 	]+foo_global
.*:[ 	]+0000000000000000[ 	]+jvt.jt\[1\]:.*
#...
.*:[ 	]+0000000000000000[ 	]+jvt.jt\[31\]:.*
.*:[ 	]+[0-9a-f]+[ 	]+jvt.jalt\[32\]:[ 	]+foo_global
#...
Disassembly of section .text:
0+[0-9a-f]+ <_start>:
.*:[ 	]+[0-9a-f]+[ 	]+cm.jt[ 	]+0 # [0-9a-f]+ <foo_global>
.*:[ 	]+[0-9a-f]+[ 	]+cm.jt[ 	]+0 # [0-9a-f]+ <foo_global>
.*:[ 	]+[0-9a-f]+[ 	]+cm.jt[ 	]+0 # [0-9a-f]+ <foo_global>
.*:[ 	]+[0-9a-f]+[ 	]+cm.jt[ 	]+0 # [0-9a-f]+ <foo_global>
.*:[ 	]+[0-9a-f]+[ 	]+cm.jt[ 	]+0 # [0-9a-f]+ <foo_global>
.*:[ 	]+[0-9a-f]+[ 	]+cm.jalt[ 	]+32 # [0-9a-f]+ <foo_global>
.*:[ 	]+[0-9a-f]+[ 	]+cm.jalt[ 	]+32 # [0-9a-f]+ <foo_global>
.*:[ 	]+[0-9a-f]+[ 	]+cm.jalt[ 	]+32 # [0-9a-f]+ <foo_global>
.*:[ 	]+[0-9a-f]+[ 	]+cm.jalt[ 	]+32 # [0-9a-f]+ <foo_global>
#...
