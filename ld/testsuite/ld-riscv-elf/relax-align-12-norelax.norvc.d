#source: relax-align-12.s
#as: -mrelax -march=rv64gc -defsym NORVC=1 -defsym NORELAX=1
#ld: -melf64lriscv -Trelax-align.ld
#objdump: -d

.*:[ ]+file format .*


Disassembly of section .text:

0+1000 <_start>:
[ 	]+1000:[ 	]+014000ef[ 	]+jal[ 	]+[0-9a-z]+[ 	]+<foo>
[ 	]+1004:[ 	]+02e58533[ 	]+mul[ 	]+a0,a1,a4
[ 	]+1008:[ 	]+00000013[ 	]+nop
[ 	]+100c:[ 	]+00000013[ 	]+nop

0+1010 <SHOULD_ALIGN_16_HERE>:
[ 	]+1010:[ 	]+00008067[ 	]+ret

0+1014 <foo>:
[ 	]+1014:[ 	]+00008067[ 	]+ret
[ 	]+1018:[ 	]+0001[ 	]+\.insn[ 	]+2, 0x0001
[ 	]+101a:[ 	]+00000013[ 	]+nop
[ 	]+101e:[ 	]+00000013[ 	]+nop
[ 	]+1022:[ 	]+00000013[ 	]+nop
