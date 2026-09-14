#as: -march=rv64i_xxtvcoder
#objdump: -dr

.*:[ 	]+file format .*

Disassembly of section .text:

0+000 <.text>:
[ 	]+[0-9a-f]+:[ 	]+c211700b[ 	]+xt.vabd.vv[ 	]+v0,v1,v2
[ 	]+[0-9a-f]+:[ 	]+c615700b[ 	]+xt.vabd.vx[ 	]+v0,v1,a0
[ 	]+[0-9a-f]+:[ 	]+ca12f00b[ 	]+xt.vabd.vi[ 	]+v0,v1,5
[ 	]+[0-9a-f]+:[ 	]+da11700b[ 	]+xt.vaba.vv[ 	]+v0,v1,v2
[ 	]+[0-9a-f]+:[ 	]+de15700b[ 	]+xt.vaba.vx[ 	]+v0,v1,a0
[ 	]+[0-9a-f]+:[ 	]+ce12700b[ 	]+xt.vaba.vi[ 	]+v0,v1,4
[ 	]+[0-9a-f]+:[ 	]+a211600b[ 	]+xt.vabdu.vv[ 	]+v0,v1,v2
[ 	]+[0-9a-f]+:[ 	]+a615600b[ 	]+xt.vabdu.vx[ 	]+v0,v1,a0
[ 	]+[0-9a-f]+:[ 	]+aa12600b[ 	]+xt.vabdu.vi[ 	]+v0,v1,4
[ 	]+[0-9a-f]+:[ 	]+b211600b[ 	]+xt.vabau.vv[ 	]+v0,v1,v2
[ 	]+[0-9a-f]+:[ 	]+b615600b[ 	]+xt.vabau.vx[ 	]+v0,v1,a0
[ 	]+[0-9a-f]+:[ 	]+ae12600b[ 	]+xt.vabau.vi[ 	]+v0,v1,4
[ 	]+[0-9a-f]+:[ 	]+e211700b[ 	]+xt.vwabd.vv[ 	]+v0,v1,v2
[ 	]+[0-9a-f]+:[ 	]+e616700b[ 	]+xt.vwabd.vx[ 	]+v0,v1,a2
[ 	]+[0-9a-f]+:[ 	]+f211700b[ 	]+xt.vwaba.vv[ 	]+v0,v1,v2
[ 	]+[0-9a-f]+:[ 	]+f616700b[ 	]+xt.vwaba.vx[ 	]+v0,v1,a2
[ 	]+[0-9a-f]+:[ 	]+fa11700b[ 	]+xt.vwabau.vv[ 	]+v0,v1,v2
[ 	]+[0-9a-f]+:[ 	]+fe16700b[ 	]+xt.vwabau.vx[ 	]+v0,v1,a2
[ 	]+[0-9a-f]+:[ 	]+ea11700b[ 	]+xt.vwabdu.vv[ 	]+v0,v1,v2
[ 	]+[0-9a-f]+:[ 	]+ee16700b[ 	]+xt.vwabdu.vx[ 	]+v0,v1,a2
[ 	]+[0-9a-f]+:[ 	]+d211700b[ 	]+xt.vfabd.vv[ 	]+v0,v1,v2
[ 	]+[0-9a-f]+:[ 	]+d616700b[ 	]+xt.vfabd.vf[ 	]+v0,v1,fa2
[ 	]+[0-9a-f]+:[ 	]+6a21f08b[ 	]+xt.vabsmax.vv[ 	]+v1,v2,v3
[ 	]+[0-9a-f]+:[ 	]+6821f08b[ 	]+xt.vabsmax.vv[ 	]+v1,v2,v3,v0.t
