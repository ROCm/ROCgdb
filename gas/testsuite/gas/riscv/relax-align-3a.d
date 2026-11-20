#as: -mrelax -defsym NORELAX=1 -defsym NORVC=1 -march=rv64gc
#source: relax-align-3.s
#objdump: -dr

.*:[ ]+file format .*


Disassembly of section .text:

0+0000 <_start-0x2>:
[ 	]+0:[ 	]+0001[ 	]+nop
[ 	]+0:[ 	]+R_RISCV_ALIGN[ 	]+\*ABS\*\+0x2

0+0002 <_start>:
[ 	]+2:[ 	]+00000537[ 	]+lui[ 	]+a0,0x0
[ 	]+2:[ 	]+R_RISCV_HI20[ 	]+foo
[ 	]+2:[ 	]+R_RISCV_RELAX[ 	]+\*ABS\*
[ 	]+6:[ 	]+00050513[ 	]+mv[ 	]+a0,a0
[ 	]+6:[ 	]+R_RISCV_LO12_I[ 	]+foo
[ 	]+6:[ 	]+R_RISCV_RELAX[ 	]+\*ABS\*
[ 	]+a:[ 	]+8082[ 	]+ret
[ 	]+c:[ 	]+0001[ 	]+nop
[ 	]+c:[ 	]+R_RISCV_ALIGN[ 	]+\*ABS\*\+0x6
[ 	]+e:[ 	]+00000013[ 	]+nop

0+0012 <SHOULD_ALIGN_8_HERE>:
[ 	]+12:[ 	]+12345678[ 	]+\.word[ 	]+0x12345678

0+0016 <foo>:
[ 	]+16:[ 	]+8082[ 	]+ret
