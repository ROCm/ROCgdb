#as: -march=rv32i_xandesvsinth
#source: x-andes-vsinth.s
#objdump: -d

.*:[ 	]+file format .*


Disassembly of section .text:

0+000 <target>:
[ 	]+[0-9a-f]+:[ 	]+0605425b[ 	]+nds\.vle4\.v[ 	]+v4,\(a0\)
[ 	]+[0-9a-f]+:[ 	]+0605425b[ 	]+nds\.vle4\.v[ 	]+v4,\(a0\)
[ 	]+[0-9a-f]+:[ 	]+0282425b[ 	]+nds\.vfwcvt\.f\.n\.v[ 	]+v4,v8
[ 	]+[0-9a-f]+:[ 	]+0082425b[ 	]+nds\.vfwcvt\.f\.n\.v[ 	]+v4,v8,v0\.t
[ 	]+[0-9a-f]+:[ 	]+0282c25b[ 	]+nds\.vfwcvt\.f\.nu\.v[ 	]+v4,v8
[ 	]+[0-9a-f]+:[ 	]+0082c25b[ 	]+nds\.vfwcvt\.f\.nu\.v[ 	]+v4,v8,v0\.t
[ 	]+[0-9a-f]+:[ 	]+0283425b[ 	]+nds\.vfwcvt\.f\.b\.v[ 	]+v4,v8
[ 	]+[0-9a-f]+:[ 	]+0083425b[ 	]+nds\.vfwcvt\.f\.b\.v[ 	]+v4,v8,v0\.t
[ 	]+[0-9a-f]+:[ 	]+0283c25b[ 	]+nds\.vfwcvt\.f\.bu\.v[ 	]+v4,v8
[ 	]+[0-9a-f]+:[ 	]+0083c25b[ 	]+nds\.vfwcvt\.f\.bu\.v[ 	]+v4,v8,v0\.t
