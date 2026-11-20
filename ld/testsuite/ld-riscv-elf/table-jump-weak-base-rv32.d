#source: table-jump-weak-base.s
#as: -march=rv32ic_zcmt -mabi=ilp32 --defsym SITES=3
#ld: -melf32lriscv -Ttable-jump-weak-base.ld --relax-tbljal
#objdump: -d

#...
Disassembly of section .riscv.jvt:
0+1000 <__jvt_base\$>:
.*jvt.jt\[0\]:[ 	]+foo_local
.*jvt.jt\[1\]:[ 	]+foo_global
#...
Disassembly of section .text:
0+3000 <_start>:
[ 	]+3000:[ 	]+ffffe517[ 	]+auipc[ 	]+a0,0xffffe
[ 	]+3004:[ 	]+00050513[ 	]+mv[ 	]+a0,a0.*
#...
.*csrw[ 	]+jvt,a0
#...
.*cm.jt[ 	]+0.*
.*cm.jt[ 	]+1.*
#pass
