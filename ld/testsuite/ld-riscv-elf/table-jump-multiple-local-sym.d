#source: table-jump-multiple-local-sym.s
#source: table-jump-multiple-local-sym-2.s
#as: -march=rv64ic_zcmt
#ld: -Ttable-jump.ld --relax-tbljal
#objdump: -d

#...
Disassembly of section .riscv.jvt:
0+[0-9a-f]+ <__jvt_base\$>:
[ 	]+...
.*:[ 	]+[0-9a-f]+[ 	]+jvt.jalt\[32\]:[ 	]+foo_global
.*:[ 	]+[0-9a-f]+[ 	]+jvt.jalt\[33\]:[ 	]+foo_local
.*:[ 	]+[0-9a-f]+[ 	]+jvt.jalt\[34\]:[ 	]+foo_local

Disassembly of section .text:
#...
