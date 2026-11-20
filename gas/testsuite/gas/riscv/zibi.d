#as: -march=rv32i_zibi
#objdump: -d

.*:[ ]+file format .*


Disassembly of section .text:

0+[0-9a-f]+ <foo>:
.*:[ 	]+0002a063[ 	]+beqi[ 	]+t0,-1,0 <foo>
.*:[ 	]+fe5eaee3[ 	]+beqi[ 	]+t4,5,0 <foo>
.*:[ 	]+fef3ace3[ 	]+beqi[ 	]+t2,15,0 <foo>
.*:[ 	]+fffe2ae3[ 	]+beqi[ 	]+t3,31,0 <foo>
.*:[ 	]+fe02b8e3[ 	]+bnei[ 	]+t0,-1,0 <foo>
.*:[ 	]+fe5eb6e3[ 	]+bnei[ 	]+t4,5,0 <foo>
.*:[ 	]+fef3b4e3[ 	]+bnei[ 	]+t2,15,0 <foo>
.*:[ 	]+fffe32e3[ 	]+bnei[ 	]+t3,31,0 <foo>
