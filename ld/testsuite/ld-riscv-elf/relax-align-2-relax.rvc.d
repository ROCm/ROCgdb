#source: relax-align-2.s
#as: -mrelax -march=rv64gc
#ld: -melf64lriscv -Trelax-align.ld
#objdump: -d

.*:[ ]+file format .*


Disassembly of section .text:

0+1000 <_start>:
[ 	]+1000:[ 	]+00c000ef[ 	]+jal[ 	]+100c <foo>
[ 	]+1004:[ 	]+02e58533[ 	]+mul[ 	]+a0,a1,a4

0+1008 <SHOULD_ALIGN_8_HERE>:
[ 	]+1008:[ 	]+12345678[ 	]+\.word[ 	]+0x12345678

0+100c <foo>:
[ 	]+100c:[ 	]+8082[ 	]+ret
[ 	]+100e:[ 	]+0001[ 	]+nop
[ 	]+1010:[ 	]+00000013[ 	]+nop
