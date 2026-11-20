#source: relax-align-11.s
#as: -mrelax -march=rv64gc
#ld: -melf64lriscv -Trelax-align.ld
#objdump: -d

.*:[ ]+file format .*


Disassembly of section .text:

0+1000 <_start>:
[ 	]+1000:[ 	]+6505[ 	]+lui[ 	]+a0,0x1
[ 	]+1002:[ 	]+01450513[ 	]+addi[ 	]+a0,a0,20[ 	]+#.*<foo>
[ 	]+1006:[ 	]+02e58533[ 	]+mul[ 	]+a0,a1,a4
[ 	]+100a:[ 	]+00000013[ 	]+nop
[ 	]+100e:[ 	]+0001[ 	]+\.insn[ 	]+2, 0x0001

0+1010 <SHOULD_ALIGN_16_HERE>:
[ 	]+1010:[ 	]+00008067[ 	]+ret

0+1014 <foo>:
[ 	]+1014:[ 	]+00008067[ 	]+ret
[ 	]+1018:[ 	]+0001[ 	]+\.insn[ 	]+2, 0x0001
[ 	]+101a:[ 	]+00000013[ 	]+nop
[ 	]+101e:[ 	]+00000013[ 	]+nop
[ 	]+1022:[ 	]+00000013[ 	]+nop
