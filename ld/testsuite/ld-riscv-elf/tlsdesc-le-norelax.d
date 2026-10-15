#source: tlsdesc-le.s
#ld: -no-pie --no-relax
#objdump: -d --no-show-raw-insn

.*:[ 	]+file format .*


Disassembly of section .text:

0+[0-9a-f]+ <_start>:
[ 	]+[0-9a-f]+:[ 	]+nop
[ 	]+[0-9a-f]+:[ 	]+nop
[ 	]+[0-9a-f]+:[ 	]+lui[ 	]+a0,0x1
[ 	]+[0-9a-f]+:[ 	]+addi[ 	]+a0,a0,564 # 1234 <sl>
[ 	]+[0-9a-f]+:[ 	]+add[ 	]+a0,a0,tp
[ 	]+[0-9a-f]+:[ 	]+ret
