#as: -march=rv32i
#objdump: -d

.*:[ 	]+file format .*

Disassembly of section .text:

0+000 <eew32>:
[ 	]+0:[ 	]+020ff007[ 	]+\.insn[ 	]+4, ?0x020ff007

0+004 <eew64>:
[ 	]+4:[ 	]+020ff007[ 	]+vle64\.v[ 	]+v0,\(t6\)
#pass
