#as: -march=rv32i_xandesvpackfph
#source: x-andes-vpackfph.s
#objdump: -d

.*:[ 	]+file format .*


Disassembly of section .text:

0+000 <target>:
[ 	]+[0-9a-f]+:[ 	]+0a86425b[ 	]+nds\.vfpmadt\.vf[ 	]+v4,fa2,v8
[ 	]+[0-9a-f]+:[ 	]+0886425b[ 	]+nds\.vfpmadt\.vf[ 	]+v4,fa2,v8,v0\.t
[ 	]+[0-9a-f]+:[ 	]+0e86425b[ 	]+nds\.vfpmadb\.vf[ 	]+v4,fa2,v8
[ 	]+[0-9a-f]+:[ 	]+0c86425b[ 	]+nds\.vfpmadb\.vf[ 	]+v4,fa2,v8,v0\.t
