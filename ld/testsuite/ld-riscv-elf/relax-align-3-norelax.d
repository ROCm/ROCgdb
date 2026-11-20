#source: relax-align-3.s
#as: -mrelax -march=rv64gc -defsym NORELAX=1
#ld: -melf64lriscv -Trelax-align.ld
#objdump: -d

.*:[ ]+file format .*


Disassembly of section .text:

0+1000 <_start>:
[ 	]+1000:[ 	]+6505[ 	]+lui[ 	]+a0,0x1
[ 	]+1002:[ 	]+01050513[ 	]+addi[ 	]+a0,a0,16[ 	]+#.*<foo>
[ 	]+1006:[ 	]+02e58533[ 	]+mul[ 	]+a0,a1,a4
[ 	]+100a:[ 	]+0001[ 	]+nop

0+100c <SHOULD_ALIGN_4_HERE>:
[ 	]+100c:[ 	]+12345678[ 	]+\.word[ 	]+0x12345678

0+1010 <foo>:
[ 	]+1010:[ 	]+8082[ 	]+ret
