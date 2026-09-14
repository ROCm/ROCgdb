#as: -march=rv64gc_zvabd
#objdump: -dr

.*:[ 	]+file format .*

Disassembly of section .text:

0+000 <target>:
[ 	]+[0-9a-f]+:[ 	]+56206257[ 	]+vabs.v[ 	]+v4,v2
[ 	]+[0-9a-f]+:[ 	]+5620a257[ 	]+vabd.vv[ 	]+v4,v2,v1
[ 	]+[0-9a-f]+:[ 	]+56256257[ 	]+vabd.vx[ 	]+v4,v2,a0
[ 	]+[0-9a-f]+:[ 	]+5a20a257[ 	]+vabdu.vv[ 	]+v4,v2,v1
[ 	]+[0-9a-f]+:[ 	]+5a256257[ 	]+vabdu.vx[ 	]+v4,v2,a0
[ 	]+[0-9a-f]+:[ 	]+f6208257[ 	]+vwabda.vv[ 	]+v4,v2,v1
[ 	]+[0-9a-f]+:[ 	]+f6254257[ 	]+vwabda.vx[ 	]+v4,v2,a0
[ 	]+[0-9a-f]+:[ 	]+fa208257[ 	]+vwabdau.vv[ 	]+v4,v2,v1
[ 	]+[0-9a-f]+:[ 	]+fa254257[ 	]+vwabdau.vx[ 	]+v4,v2,a0
