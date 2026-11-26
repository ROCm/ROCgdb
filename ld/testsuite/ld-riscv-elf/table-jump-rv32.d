#source: table-jump-rv32.s
#as: -march=rv32ic_zcmt
#ld: -Ttable-jump.ld --relax-tbljal -m[riscv_choose_ilp32_emul]
#objdump: -d
#...
Disassembly of section .riscv.jvt:
0+[0-9a-f]+ <__jvt_base\$>:
.*:[ 	]+[0-9a-f]+[ 	]+jvt.jt\[0\]:[ 	]+foo_local
.*:[ 	]+[0-9a-f]+[ 	]+jvt.jt\[1\]:[ 	]+foo_global
#...
Disassembly of section .text:
0+[0-9a-f]+ <_start>:
.*:[ 	]+[0-9a-f]+[ 	]+cm.jt[ 	]+0 # [0-9a-f]+ <foo_local>
.*:[ 	]+[0-9a-f]+[ 	]+cm.jt[ 	]+1 # [0-9a-f]+ <foo_global>
#pass
