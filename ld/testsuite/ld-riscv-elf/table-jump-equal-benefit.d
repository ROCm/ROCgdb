#source: table-jump-equal-benefit.s
#as: -march=rv64ic_zcmt
#ld: -Ttable-jump.ld --relax-tbljal
#objdump: -d

#...
Disassembly of section .riscv.jvt:
0+[0-9a-f]+ <__jvt_base\$>:
[ 	]+...
.*:[ 	]+[0-9a-f]+[ 	]+jvt.jalt\[32\]:[ 	]+func_b
.*:[ 	]+[0-9a-f]+[ 	]+jvt.jalt\[33\]:[ 	]+func_a
#...
Disassembly of section .text:
#...
.*:[ 	]+[0-9a-f]+[ 	]+cm.jalt[ 	]+32 # [0-9a-f]+ <func_b>
#...
.*:[ 	]+[0-9a-f]+[ 	]+cm.jalt[ 	]+33 # [0-9a-f]+ <func_a>
#pass
