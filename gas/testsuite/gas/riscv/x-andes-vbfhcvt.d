#as: -march=rv32i_xandesvbfhcvt
#source: x-andes-vbfhcvt.s
#objdump: -d

.*:[ 	]+file format .*


Disassembly of section .text:

0+000 <target>:
[ 	]+[0-9a-f]+:[ 	]+0080425b[ 	]+nds\.vfwcvt\.s\.bf16[ 	]+v4,v8
[ 	]+[0-9a-f]+:[ 	]+0080c25b[ 	]+nds\.vfncvt\.bf16\.s[ 	]+v4,v8
