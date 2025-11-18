#as: -march=rv32i_xandesvsintload
#source: x-andes-vsintload.s
#objdump: -d

.*:[ 	]+file format .*


Disassembly of section .text:

0+000 <target>:
[ 	]+[0-9a-f]+:[ 	]+0625425b[ 	]+nds\.vln8\.v[  	]+v4,\(a0\)
[ 	]+[0-9a-f]+:[ 	]+0625425b[ 	]+nds\.vln8\.v[  	]+v4,\(a0\)
[ 	]+[0-9a-f]+:[ 	]+0425425b[ 	]+nds\.vln8\.v[  	]+v4,\(a0\),v0\.t
[ 	]+[0-9a-f]+:[ 	]+0635425b[ 	]+nds\.vlnu8\.v[ 	]+v4,\(a0\)
[ 	]+[0-9a-f]+:[ 	]+0635425b[ 	]+nds\.vlnu8\.v[ 	]+v4,\(a0\)
[ 	]+[0-9a-f]+:[ 	]+0435425b[ 	]+nds\.vlnu8\.v[ 	]+v4,\(a0\),v0\.t
