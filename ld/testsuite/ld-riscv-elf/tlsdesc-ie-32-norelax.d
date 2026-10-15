#source: tlsdesc-ie-32.s
#as: -march=rv32i -mabi=ilp32
#ld: -melf32lriscv -no-pie --no-relax tmpdir/tlsdesc-lib32.so
#objdump: -d --no-show-raw-insn

.*:[ 	]+file format .*


Disassembly of section .text:

0+[0-9a-f]+ <_start>:
[ 	]+[0-9a-f]+:[ 	]+nop
#?0+[0-9a-f]+ <_PROCEDURE_LINKAGE_TABLE_>:
[ 	]+[0-9a-f]+:[ 	]+nop
#...
[ 	]+[0-9a-f]+:[ 	]+auipc[ 	]+a0,0x[0-9a-f]+
#?0+[0-9a-f]+ <_PROCEDURE_LINKAGE_TABLE_>:
[ 	]+[0-9a-f]+:[ 	]+lw[ 	]+a0,-?[0-9]+\(a0\) # [0-9a-f]+ <sg2>
[ 	]+[0-9a-f]+:[ 	]+add[ 	]+a0,a0,tp
#pass
