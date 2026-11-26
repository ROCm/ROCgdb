#source: table-jump-weak-base.s
#as: -march=rv32ic_zcmt -mabi=ilp32 --defsym SITES=1
#ld: -m[riscv_choose_ilp32_emul] -Ttable-jump-weak-base.ld --relax-tbljal
#objdump: -d

#...
Disassembly of section .text:
0+3000 <_start>:
[ 	]+3000:[ 	]+ffffd517[ 	]+auipc[ 	]+a0,0xffffd
[ 	]+3004:[ 	]+00050513[ 	]+mv[ 	]+a0,a0.*
#...
.*csrw[ 	]+jvt,a0
#...
0+[0-9a-f]+ <entry>:
.*:[ 	]+[0-9a-f]+[ 	]+j[ 	]+[0-9a-f]+ <foo_local>
.*:[ 	]+[0-9a-f]+[ 	]+j[ 	]+[0-9a-f]+ <foo_global>
#pass
