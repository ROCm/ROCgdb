#as: -march=rv32i_xandesbfhcvt
#source: x-andes-bfhcvt.s
#objdump: -d

.*:[ 	]+file format .*


Disassembly of section .text:

0+000 <target>:
[ 	]+[0-9a-f]+:[ 	]+00b1455b[ 	]+nds\.fcvt\.s\.bf16[ 	]+fa0,fa1
[ 	]+[0-9a-f]+:[ 	]+00b1c55b[ 	]+nds\.fcvt\.bf16\.s[ 	]+fa0,fa1
