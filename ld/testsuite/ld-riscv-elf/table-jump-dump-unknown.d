#source: table-jump.s
#as: -march=rv64ic_zcmt --defsym TEST_CM_JT=1
#ld: -Ttable-jump.ld --strip-all --relax-tbljal
#objdump: -d
#...
Disassembly of section .riscv.jvt:
0+[0-9a-f]+ <.riscv.jvt>:
.*:[ 	]+[0-9a-f]+[ 	]+jvt.jt\[0\]:[ 	]+<unknown>
.*:[ 	]+[0-9a-f]+[ 	]+jvt.jt\[1\]:[ 	]+<unknown>
#...
Disassembly of section .text:
0+[0-9a-f]+ <.text>:
.*:[ 	]+[0-9a-f]+[ 	]+cm.jt[ 	]+0
.*:[ 	]+[0-9a-f]+[ 	]+cm.jt[ 	]+1
.*:[ 	]+[0-9a-f]+[ 	]+cm.jt[ 	]+0
.*:[ 	]+[0-9a-f]+[ 	]+cm.jt[ 	]+1
.*:[ 	]+[0-9a-f]+[ 	]+cm.jt[ 	]+0
.*:[ 	]+[0-9a-f]+[ 	]+cm.jt[ 	]+1
.*:[ 	]+[0-9a-f]+[ 	]+cm.jt[ 	]+0
.*:[ 	]+[0-9a-f]+[ 	]+cm.jt[ 	]+1
.*:[ 	]+[0-9a-f]+[ 	]+cm.jt[ 	]+0
.*:[ 	]+[0-9a-f]+[ 	]+cm.jt[ 	]+1
#...
.*:[ 	]+[0-9a-f]+[ 	]+ret
.*:[ 	]+[0-9a-f]+[ 	]+ret
