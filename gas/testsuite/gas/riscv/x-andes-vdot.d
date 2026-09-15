#as: -march=rv32i_xandesvdot
#source: x-andes-vdot.s
#objdump: -d

.*:[ 	]+file format .*


Disassembly of section .text:

0+000 <target>:
[ 	]+[0-9a-f]+:[ 	]+12c4425b[ 	]+nds\.vd4dots\.vv[ 	]+v4,v8,v12
[ 	]+[0-9a-f]+:[ 	]+10c4425b[ 	]+nds\.vd4dots\.vv[ 	]+v4,v8,v12,v0\.t
[ 	]+[0-9a-f]+:[ 	]+1ec4425b[ 	]+nds\.vd4dotu\.vv[ 	]+v4,v8,v12
[ 	]+[0-9a-f]+:[ 	]+1cc4425b[ 	]+nds\.vd4dotu\.vv[ 	]+v4,v8,v12,v0\.t
[ 	]+[0-9a-f]+:[ 	]+16c4425b[ 	]+nds\.vd4dotsu\.vv[ 	]+v4,v8,v12
[ 	]+[0-9a-f]+:[ 	]+14c4425b[ 	]+nds\.vd4dotsu\.vv[ 	]+v4,v8,v12,v0\.t
