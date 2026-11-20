#objdump: -dw -Mintel
#name: i386 AVX10_VPMLDSABFLYDQ insns (Intel disassembly)
#source: avx10-vpmldsabflydq.s

.*: +file format .*

Disassembly of section \.text:

#...
[a-f0-9]+ <_intel>:
[ 	]*[a-f0-9]+:[ 	]*62 f3 d5 08 dc f4 7b[ 	]*vpmldsabflydq xmm6,xmm5,xmm4,0x7b
[ 	]*[a-f0-9]+:[ 	]*62 f3 d5 28 dc f4 7b[ 	]*vpmldsabflydq ymm6,ymm5,ymm4,0x7b
[ 	]*[a-f0-9]+:[ 	]*62 f3 d5 48 dc f4 7b[ 	]*vpmldsabflydq zmm6,zmm5,zmm4,0x7b
[ 	]*[a-f0-9]+:[ 	]*62 f3 d5 08 dc b4 f4 00 00 00 10 7b[ 	]*vpmldsabflydq xmm6,xmm5,XMMWORD PTR \[esp\+esi\*8\+0x10000000\],0x7b
[ 	]*[a-f0-9]+:[ 	]*62 f3 d5 08 dc 31 7b[ 	]*vpmldsabflydq xmm6,xmm5,XMMWORD PTR \[ecx\],0x7b
[ 	]*[a-f0-9]+:[ 	]*62 f3 d5 08 dc 71 7f 7b[ 	]*vpmldsabflydq xmm6,xmm5,XMMWORD PTR \[ecx\+0x7f0\],0x7b
[ 	]*[a-f0-9]+:[ 	]*62 f3 d5 08 dc 72 80 7b[ 	]*vpmldsabflydq xmm6,xmm5,XMMWORD PTR \[edx-0x800\],0x7b
[ 	]*[a-f0-9]+:[ 	]*62 f3 d5 28 dc b4 f4 00 00 00 10 7b[ 	]*vpmldsabflydq ymm6,ymm5,YMMWORD PTR \[esp\+esi\*8\+0x10000000\],0x7b
[ 	]*[a-f0-9]+:[ 	]*62 f3 d5 28 dc 31 7b[ 	]*vpmldsabflydq ymm6,ymm5,YMMWORD PTR \[ecx\],0x7b
[ 	]*[a-f0-9]+:[ 	]*62 f3 d5 28 dc 71 7f 7b[ 	]*vpmldsabflydq ymm6,ymm5,YMMWORD PTR \[ecx\+0xfe0\],0x7b
[ 	]*[a-f0-9]+:[ 	]*62 f3 d5 28 dc 72 80 7b[ 	]*vpmldsabflydq ymm6,ymm5,YMMWORD PTR \[edx-0x1000\],0x7b
[ 	]*[a-f0-9]+:[ 	]*62 f3 d5 48 dc b4 f4 00 00 00 10 7b[ 	]*vpmldsabflydq zmm6,zmm5,ZMMWORD PTR \[esp\+esi\*8\+0x10000000\],0x7b
[ 	]*[a-f0-9]+:[ 	]*62 f3 d5 48 dc 31 7b[ 	]*vpmldsabflydq zmm6,zmm5,ZMMWORD PTR \[ecx\],0x7b
[ 	]*[a-f0-9]+:[ 	]*62 f3 d5 48 dc 71 7f 7b[ 	]*vpmldsabflydq zmm6,zmm5,ZMMWORD PTR \[ecx\+0x1fc0\],0x7b
[ 	]*[a-f0-9]+:[ 	]*62 f3 d5 48 dc 72 80 7b[ 	]*vpmldsabflydq zmm6,zmm5,ZMMWORD PTR \[edx-0x2000\],0x7b
#pass
