#source: tlsdesc-ie.s
#ld: -no-pie --no-relax tmpdir/tlsdesc-lib.so
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
[ 	]+[0-9a-f]+:[ 	]+ld[ 	]+a0,-?[0-9]+\(a0\) # [0-9a-f]+ <sg2>
[ 	]+[0-9a-f]+:[ 	]+add[ 	]+a0,a0,tp
#pass
