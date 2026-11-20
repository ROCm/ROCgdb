#source: table-jump-vma-zero.s
#as: -march=rv64ic_zcmt
#ld: -Ttable-jump-vma-zero.ld --relax-tbljal
#objdump: -d -s -j .riscv.jvt -j .text

#...
Contents of section .riscv.jvt:
[ 	]+1000[ 	]+00000000[ 	]+00000000.*
#...
Disassembly of section .text:
#...
0+3000 <_start>:
.*:[ 	]+[0-9a-f]+[ 	]+cm.jt[ 	]+0 # 0 <target>
.*:[ 	]+[0-9a-f]+[ 	]+cm.jt[ 	]+0 # 0 <target>
#pass
