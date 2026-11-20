#source: table-jump.s
#as: -march=rv64ic_zcmt --defsym TEST_CM_JALT=1
#ld: -Ttable-jump-vma-lma.ld --relax-tbljal
#objdump: -d

#...
Disassembly of section .riscv.jvt:
#...
.*:[ 	]+0000000000003118[ 	]+jvt.jalt\[32\]:[ 	]+foo_global
.*:[ 	]+000000000000311a[ 	]+jvt.jalt\[33\]:[ 	]+foo_local
#...
Disassembly of section .text:
#...
.*:[ 	]+[0-9a-f]+[ 	]+cm.jalt[ 	]+32 # 3118 <foo_global>
.*:[ 	]+[0-9a-f]+[ 	]+cm.jalt[ 	]+33 # 311a <foo_local>
#pass
