#source: relax-align-8.s
#as: -mrelax -march=rv64gc -defsym NORELAX=1
#ld: -melf64lriscv -Trelax-align.ld
#objdump: -d

.*:[ ]+file format .*


Disassembly of section .text:

0+1000 <_start>:
[ 	]+1000:[ 	]+00c000ef[ 	]+jal[ 	]+100c <foo>
[ 	]+1004:[ 	]+02e58533[ 	]+mul[ 	]+a0,a1,a4

0+1008 <SHOULD_ALIGN_4_HERE>:
[ 	]+1008:[ 	]+12345678[ 	]+\.word[ 	]+0x12345678

0+100c <foo>:
[ 	]+100c:[ 	]+00008067[ 	]+ret
[ 	]+1010:[ \t]+0001[ \t]+\.insn\t2, 0x0001
