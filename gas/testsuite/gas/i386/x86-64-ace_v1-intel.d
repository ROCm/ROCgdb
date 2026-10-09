#as: -march=generic64+ace_v1
#objdump: -dw -Mintel
#name: 64-bit ACE V1 insns (Intel disassembly)
#source: x86-64-ace_v1.s

.*: +file format .*

Disassembly of section \.text:

[0-9a-f]+ <start>:
[ 	]*[a-f0-9]+:[ 	]*c4 e2 7b 49 c0[ 	]+tilezero tmm0
[ 	]*[a-f0-9]+:[ 	]*c4 e2 78 49 01[ 	]+ldtilecfg \[rcx\]
[ 	]*[a-f0-9]+:[ 	]*c4 e2 79 49 01[ 	]+sttilecfg \[rcx\]
[ 	]*[a-f0-9]+:[ 	]*c4 e2 78 49 c0[ 	]+tilerelease
[ 	]*[a-f0-9]+:[ 	]*62 f2 65 48 4a ca[ 	]+tilemovrow zmm1,tmm2,ebx
[ 	]*[a-f0-9]+:[ 	]*62 f3 7d 48 07 ca 08[ 	]+tilemovrow zmm1,tmm2,0x8
[ 	]*[a-f0-9]+:[ 	]*62 f2 e5 48 4a ca[ 	]+tilemovrow tmm1,zmm2,ebx
[ 	]*[a-f0-9]+:[ 	]*62 f3 fd 48 07 ca 08[ 	]+tilemovrow tmm1,zmm2,0x8
[ 	]*[a-f0-9]+:[ 	]*62 f2 e5 48 4b ca[ 	]+tilemovcol tmm1,zmm2,ebx
[ 	]*[a-f0-9]+:[ 	]*62 f3 fd 48 2f ca 08[ 	]+tilemovcol tmm1,zmm2,0x8
[ 	]*[a-f0-9]+:[ 	]*62 62 6e 48 4a f5[ 	]+tcvtrowd2ps zmm30,tmm5,edx
[ 	]*[a-f0-9]+:[ 	]*62 63 7e 48 07 f5 08[ 	]+tcvtrowd2ps zmm30,tmm5,0x8
[ 	]*[a-f0-9]+:[ 	]*62 62 6f 48 6d f5[ 	]+tcvtrowps2bf16h zmm30,tmm5,edx
[ 	]*[a-f0-9]+:[ 	]*62 63 7f 48 07 f5 08[ 	]+tcvtrowps2bf16h zmm30,tmm5,0x8
[ 	]*[a-f0-9]+:[ 	]*62 62 6c 48 6d f5[ 	]+tcvtrowps2phh zmm30,tmm5,edx
[ 	]*[a-f0-9]+:[ 	]*62 63 7c 48 07 f5 08[ 	]+tcvtrowps2phh zmm30,tmm5,0x8
[ 	]*[a-f0-9]+:[ 	]*62 d6 ff 48 95 c2[ 	]+bsrmovh bsr0,zmm10
[ 	]*[a-f0-9]+:[ 	]*62 f6 ff 48 95 41 7f[ 	]+bsrmovh bsr0,ZMMWORD PTR \[rcx\+0x1fc0\]
[ 	]*[a-f0-9]+:[ 	]*62 f6 7f 48 95 c4[ 	]+bsrmovh zmm4,bsr0
[ 	]*[a-f0-9]+:[ 	]*62 f6 7f 48 95 41 80[ 	]+bsrmovh ZMMWORD PTR \[rcx-0x2000\],bsr0
[ 	]*[a-f0-9]+:[ 	]*62 62 6e 48 6d f5[ 	]+tcvtrowps2bf16l zmm30,tmm5,edx
[ 	]*[a-f0-9]+:[ 	]*62 63 7e 48 77 f5 08[ 	]+tcvtrowps2bf16l zmm30,tmm5,0x8
[ 	]*[a-f0-9]+:[ 	]*62 62 6d 48 6d f5[ 	]+tcvtrowps2phl zmm30,tmm5,edx
[ 	]*[a-f0-9]+:[ 	]*62 63 7f 48 77 f5 08[ 	]+tcvtrowps2phl zmm30,tmm5,0x8
[ 	]*[a-f0-9]+:[ 	]*62 d6 fe 48 95 c2[ 	]+bsrmovl bsr0,zmm10
[ 	]*[a-f0-9]+:[ 	]*62 f6 fe 48 95 41 7f[ 	]+bsrmovl bsr0,ZMMWORD PTR \[rcx\+0x1fc0\]
[ 	]*[a-f0-9]+:[ 	]*62 f6 7e 48 95 c4[ 	]+bsrmovl zmm4,bsr0
[ 	]*[a-f0-9]+:[ 	]*62 f6 7e 48 95 41 80[ 	]+bsrmovl ZMMWORD PTR \[rcx-0x2000\],bsr0
[ 	]*[a-f0-9]+:[ 	]*c4 e2 fb 49 c0[ 	]+bsrinit bsr0
[ 	]*[a-f0-9]+:[ 	]*62 f6 f4 48 95 c3[ 	]+bsrmovf bsr0,zmm1,zmm3
[ 	]*[a-f0-9]+:[ 	]*62 f6 f4 48 95 41 7f[ 	]+bsrmovf bsr0,zmm1,ZMMWORD PTR \[rcx\+0x1fc0\]
[ 	]*[a-f0-9]+:[ 	]*62 f3 6c 48 8d c9 00[ 	]+top4mxbf8ps tmm1,zmm1,zmm2,0x0
[ 	]*[a-f0-9]+:[ 	]*62 f3 6f 48 8d c9 00[ 	]+top4mxbhf8ps tmm1,zmm1,zmm2,0x0
[ 	]*[a-f0-9]+:[ 	]*62 f3 6e 48 8d c9 00[ 	]+top4mxhbf8ps tmm1,zmm1,zmm2,0x0
[ 	]*[a-f0-9]+:[ 	]*62 f3 6d 48 8d c9 00[ 	]+top4mxhf8ps tmm1,zmm1,zmm2,0x0
[ 	]*[a-f0-9]+:[ 	]*62 f3 6f 48 8f c9 00[ 	]+top4mxbssps tmm1,zmm1,zmm2,0x0
[ 	]*[a-f0-9]+:[ 	]*62 f2 6e 48 5c c1[ 	]+top2bf16ps tmm0,zmm1,zmm2
[ 	]*[a-f0-9]+:[ 	]*62 f2 6f 48 5e c1[ 	]+top4bssd tmm0,zmm1,zmm2
[ 	]*[a-f0-9]+:[ 	]*62 f2 6e 48 5e c1[ 	]+top4bsud tmm0,zmm1,zmm2
[ 	]*[a-f0-9]+:[ 	]*62 f2 6d 48 5e c1[ 	]+top4busd tmm0,zmm1,zmm2
[ 	]*[a-f0-9]+:[ 	]*62 f2 6c 48 5e c1[ 	]+top4buud tmm0,zmm1,zmm2
#pass
