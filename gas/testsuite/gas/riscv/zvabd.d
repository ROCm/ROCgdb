#as: -march=rv64gc_zvabd
#objdump: -dr

.*:[ 	]+file format .*

Disassembly of section .text:

0+000 <target>:
[ 	]+[0-9a-f]+:[ 	]+4a282257[ 	]+vabs.v[ 	]+v4,v2
[ 	]+[0-9a-f]+:[ 	]+4620a257[ 	]+vabd.vv[ 	]+v4,v2,v1
[ 	]+[0-9a-f]+:[ 	]+4e20a257[ 	]+vabdu.vv[ 	]+v4,v2,v1
[ 	]+[0-9a-f]+:[ 	]+5620a257[ 	]+vwabda.vv[ 	]+v4,v2,v1
[ 	]+[0-9a-f]+:[ 	]+5a20a257[ 	]+vwabdau.vv[ 	]+v4,v2,v1
