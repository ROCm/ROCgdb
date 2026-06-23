#as: -march=rv64gcv_xsmtvdotii
#objdump: -dr

.*:[ 	]+file format .*


Disassembly of section .text:

0+000 <target>:
[ 	]+[0-9a-f]+:[ 	]+c241b12b[ 	]+smt.vmadot[ 	]+v2,v3,v4,i4
[ 	]+[0-9a-f]+:[ 	]+c241812b[ 	]+smt.vmadotu[ 	]+v2,v3,v4,i4
[ 	]+[0-9a-f]+:[ 	]+c241a12b[ 	]+smt.vmadotsu[ 	]+v2,v3,v4,i4
[ 	]+[0-9a-f]+:[ 	]+c241912b[ 	]+smt.vmadotus[ 	]+v2,v3,v4,i4
[ 	]+[0-9a-f]+:[ 	]+e241b12b[ 	]+smt.vmadot[ 	]+v2,v3,v4
[ 	]+[0-9a-f]+:[ 	]+e241812b[ 	]+smt.vmadotu[ 	]+v2,v3,v4
[ 	]+[0-9a-f]+:[ 	]+e241a12b[ 	]+smt.vmadotsu[ 	]+v2,v3,v4
[ 	]+[0-9a-f]+:[ 	]+e241912b[ 	]+smt.vmadotus[ 	]+v2,v3,v4
[ 	]+[0-9a-f]+:[ 	]+e241b12b[ 	]+smt.vmadot[ 	]+v2,v3,v4
[ 	]+[0-9a-f]+:[ 	]+e241812b[ 	]+smt.vmadotu[ 	]+v2,v3,v4
[ 	]+[0-9a-f]+:[ 	]+e241a12b[ 	]+smt.vmadotsu[ 	]+v2,v3,v4
[ 	]+[0-9a-f]+:[ 	]+e241912b[ 	]+smt.vmadotus[ 	]+v2,v3,v4
[ 	]+[0-9a-f]+:[ 	]+e652012b[ 	]+smt.vmadot1u[ 	]+v2,v4,v5
[ 	]+[0-9a-f]+:[ 	]+e652312b[ 	]+smt.vmadot1[ 	]+v2,v4,v5
[ 	]+[0-9a-f]+:[ 	]+e652212b[ 	]+smt.vmadot1su[ 	]+v2,v4,v5
[ 	]+[0-9a-f]+:[ 	]+e652112b[ 	]+smt.vmadot1us[ 	]+v2,v4,v5
[ 	]+[0-9a-f]+:[ 	]+e652412b[ 	]+smt.vmadot2u[ 	]+v2,v4,v5
[ 	]+[0-9a-f]+:[ 	]+e652712b[ 	]+smt.vmadot2[ 	]+v2,v4,v5
[ 	]+[0-9a-f]+:[ 	]+e652612b[ 	]+smt.vmadot2su[ 	]+v2,v4,v5
[ 	]+[0-9a-f]+:[ 	]+e652512b[ 	]+smt.vmadot2us[ 	]+v2,v4,v5
[ 	]+[0-9a-f]+:[ 	]+e652812b[ 	]+smt.vmadot3u[ 	]+v2,v4,v5
[ 	]+[0-9a-f]+:[ 	]+e652b12b[ 	]+smt.vmadot3[ 	]+v2,v4,v5
[ 	]+[0-9a-f]+:[ 	]+e652a12b[ 	]+smt.vmadot3su[ 	]+v2,v4,v5
[ 	]+[0-9a-f]+:[ 	]+e652912b[ 	]+smt.vmadot3us[ 	]+v2,v4,v5
[ 	]+[0-9a-f]+:[ 	]+c852012b[ 	]+smt.vmadotu.sp[ 	]+v2,v4,v5,v0,0,i4
[ 	]+[0-9a-f]+:[ 	]+c85201ab[ 	]+smt.vmadotu.sp[ 	]+v2,v4,v5,v0,1,i4
[ 	]+[0-9a-f]+:[ 	]+c852312b[ 	]+smt.vmadot.sp[ 	]+v2,v4,v5,v0,0,i4
[ 	]+[0-9a-f]+:[ 	]+c852212b[ 	]+smt.vmadotsu.sp[ 	]+v2,v4,v5,v0,0,i4
[ 	]+[0-9a-f]+:[ 	]+c852112b[ 	]+smt.vmadotus.sp[ 	]+v2,v4,v5,v0,0,i4
[ 	]+[0-9a-f]+:[ 	]+e852012b[ 	]+smt.vmadotu.sp[ 	]+v2,v4,v5,v0,0
[ 	]+[0-9a-f]+:[ 	]+e85201ab[ 	]+smt.vmadotu.sp[ 	]+v2,v4,v5,v0,1
[ 	]+[0-9a-f]+:[ 	]+e852312b[ 	]+smt.vmadot.sp[ 	]+v2,v4,v5,v0,0
[ 	]+[0-9a-f]+:[ 	]+e852212b[ 	]+smt.vmadotsu.sp[ 	]+v2,v4,v5,v0,0
[ 	]+[0-9a-f]+:[ 	]+e852112b[ 	]+smt.vmadotus.sp[ 	]+v2,v4,v5,v0,0
[ 	]+[0-9a-f]+:[ 	]+e852812b[ 	]+smt.vmadotu.sp[ 	]+v2,v4,v5,v0,2
[ 	]+[0-9a-f]+:[ 	]+e85281ab[ 	]+smt.vmadotu.sp[ 	]+v2,v4,v5,v0,3
[ 	]+[0-9a-f]+:[ 	]+e852b12b[ 	]+smt.vmadot.sp[ 	]+v2,v4,v5,v0,2
[ 	]+[0-9a-f]+:[ 	]+e852a12b[ 	]+smt.vmadotsu.sp[ 	]+v2,v4,v5,v0,2
[ 	]+[0-9a-f]+:[ 	]+e852912b[ 	]+smt.vmadotus.sp[ 	]+v2,v4,v5,v0,2
[ 	]+[0-9a-f]+:[ 	]+ea52812b[ 	]+smt.vmadotu.sp[ 	]+v2,v4,v5,v1,2
[ 	]+[0-9a-f]+:[ 	]+ea52b12b[ 	]+smt.vmadot.sp[ 	]+v2,v4,v5,v1,2
[ 	]+[0-9a-f]+:[ 	]+cc41812b[ 	]+smt.vmadotu.hp[ 	]+v2,v3,v4,v0,0,i4
[ 	]+[0-9a-f]+:[ 	]+cc41912b[ 	]+smt.vmadotu.hp[ 	]+v2,v3,v4,v0,1,i4
[ 	]+[0-9a-f]+:[ 	]+ec41812b[ 	]+smt.vmadotu.hp[ 	]+v2,v3,v4,v0,0
[ 	]+[0-9a-f]+:[ 	]+d041812b[ 	]+smt.vmadot.hp[ 	]+v2,v3,v4,v0,0,i4
[ 	]+[0-9a-f]+:[ 	]+d441812b[ 	]+smt.vmadotsu.hp[ 	]+v2,v3,v4,v0,0,i4
[ 	]+[0-9a-f]+:[ 	]+d841812b[ 	]+smt.vmadotus.hp[ 	]+v2,v3,v4,v0,0,i4
[ 	]+[0-9a-f]+:[ 	]+ce41812b[ 	]+smt.vmadotu.hp[ 	]+v2,v3,v4,v1,0,i4
[ 	]+[0-9a-f]+:[ 	]+9e41c12b[ 	]+smt.vfwmadot[ 	]+v2,v3,v4
[ 	]+[0-9a-f]+:[ 	]+9e52512b[ 	]+smt.vfwmadot1[ 	]+v2,v4,v5
[ 	]+[0-9a-f]+:[ 	]+9e52612b[ 	]+smt.vfwmadot2[ 	]+v2,v4,v5
[ 	]+[0-9a-f]+:[ 	]+9e52712b[ 	]+smt.vfwmadot3[ 	]+v2,v4,v5
[ 	]+[0-9a-f]+:[ 	]+6241812b[ 	]+smt.vnpack.vv[ 	]+v2,v3,v4,0
[ 	]+[0-9a-f]+:[ 	]+6241912b[ 	]+smt.vnpack.vv[ 	]+v2,v3,v4,1
[ 	]+[0-9a-f]+:[ 	]+6241c12b[ 	]+smt.vnspack.vv[ 	]+v2,v3,v4,0
[ 	]+[0-9a-f]+:[ 	]+4241812b[ 	]+smt.vnpack4.vv[ 	]+v2,v3,v4,0
[ 	]+[0-9a-f]+:[ 	]+4241c12b[ 	]+smt.vnspack4.vv[ 	]+v2,v3,v4,0
[ 	]+[0-9a-f]+:[ 	]+6641812b[ 	]+smt.vpack.vv[ 	]+v2,v3,v4,0
[ 	]+[0-9a-f]+:[ 	]+6641912b[ 	]+smt.vpack.vv[ 	]+v2,v3,v4,1
[ 	]+[0-9a-f]+:[ 	]+6641c12b[ 	]+smt.vupack.vv[ 	]+v2,v3,v4,0
