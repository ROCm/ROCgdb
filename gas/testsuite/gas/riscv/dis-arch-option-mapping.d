#as: -march=rv64gc
#source: dis-arch-option.s
#objdump: -d -M arch=rv64gcv
# A "$xrv" mapping symbol names its own ISA and wins over "arch=".
# A bare "$x" returns to the ISA given by "arch=".

.*:[ 	]+file format .*


Disassembly of section .text:

0+000 <.text>:
[ 	]+0:[ 	]+022180d7[ 	]+\.insn[ 	]+4, 0x022180d7
[ 	]+4:[ 	]+022180d7[ 	]+vadd.vv[ 	]+v1,v2,v3
