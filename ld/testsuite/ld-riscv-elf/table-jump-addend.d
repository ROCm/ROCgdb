#source: table-jump-addend.s
#as: -march=rv64ic_zcmt
#ld: -Ttable-jump.ld --relax-tbljal
#objdump: -d
#...
Disassembly of section .riscv.jvt:
0+[0-9a-f]+ <__jvt_base\$>:
[ 	]+...
.*:[ 	]+[0-9a-f]+[ 	]+jvt.jalt\[32\]:[ 	]+target
.*:[ 	]+[0-9a-f]+[ 	]+jvt.jalt\[33\]:[ 	]+target_4
.*:[ 	]+[0-9a-f]+[ 	]+jvt.jalt\[34\]:[ 	]+target_8
#...
Disassembly of section .text:
0+[0-9a-f]+ <_start>:
.*:[ 	]+[0-9a-f]+[ 	]+cm.jalt[ 	]+32 # [0-9a-f]+ <target>
.*:[ 	]+[0-9a-f]+[ 	]+cm.jalt[ 	]+33 # [0-9a-f]+ <target_4>
.*:[ 	]+[0-9a-f]+[ 	]+cm.jalt[ 	]+34 # [0-9a-f]+ <target_8>
#pass
