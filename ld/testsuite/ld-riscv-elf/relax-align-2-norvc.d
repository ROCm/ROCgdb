#source: relax-align-2.s
#as: -mrelax -march=rv64gc -defsym NORVC=1
#ld: -melf64lriscv -Trelax-align.ld
#objdump: -d

.*:[ ]+file format .*


Disassembly of section .text:

0+1000 <_start>:
[ 	]+1000:[ 	]+014000ef[ 	]+jal[ 	]+[0-9a-z]+[ 	]+<foo>
[ 	]+1004:[ 	]+02e58533[ 	]+mul[ 	]+a0,a1,a4
[ 	]+1008:[ 	]+0001[ 	]+nop
[ 	]+100a:[ 	]+0001[ 	]+nop
[ 	]+100c:[ 	]+00000013[ 	]+nop

0+1010 <SHOULD_ALIGN_8_HERE>:
[ 	]+1010:[ 	]+12345678[ 	]+\.word[ 	]+0x12345678

0+1014 <foo>:
[ 	]+1014:[ 	]+8082[ 	]+ret
[ 	]+1016:[ 	]+00000013[ 	]+nop
