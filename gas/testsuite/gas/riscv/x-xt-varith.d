#as: -march=rv64i_xxtvarith
#objdump: -dr

.*:[ 	]+file format .*

Disassembly of section .text:

0+000 <.text>:
[ 	]+[0-9a-f]+:[ 	]+da11600b[ 	]+xt.vile.vv[ 	]+v0,v1,v2
[ 	]+[0-9a-f]+:[ 	]+f211600b[ 	]+xt.vilo.vv[ 	]+v0,v1,v2
[ 	]+[0-9a-f]+:[ 	]+fa11600b[ 	]+xt.vcrcfoldr.vv[ 	]+v0,v1,v2
[ 	]+[0-9a-f]+:[ 	]+fe11600b[ 	]+xt.vcrcfoldn.vv[ 	]+v0,v1,v2
[ 	]+[0-9a-f]+:[ 	]+7a11600b[ 	]+xt.vgmulxor.vv[ 	]+v0,v1,v2
