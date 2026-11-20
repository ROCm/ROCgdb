#source: relax-align-5.s
#as: -mrelax -march=rv64gc -defsym NORVC=1
#ld: -melf64lriscv -Trelax-align.ld
#objdump: -d

.*:[ ]+file format .*


Disassembly of section .text:

0+1000 <_start>:
[ 	]+1000:[ 	]+6505[ 	]+lui[ 	]+a0,0x1
[ 	]+1002:[ 	]+01450513[ 	]+addi[ 	]+a0,a0,20[ 	]+#.*<foo>
[ 	]+1006:[ 	]+02e58533[ 	]+mul[ 	]+a0,a1,a4
[ 	]+100a:[ 	]+00000013[ 	]+nop
[ 	]+100e:[ 	]+0001[ 	]+nop

0+1010 <SHOULD_ALIGN_16_HERE>:
[ 	]+1010:[ 	]+12345678[ 	]+\.word[ 	]+0x12345678

0+1014 <foo>:
[ 	]+1014:[ 	]+8082[ 	]+ret
