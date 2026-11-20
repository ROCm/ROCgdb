#objdump: -dw -Mintel
#name: x86_64 AVX10_VPMLDSABFLYDQ insns (Intel disassembly)
#source: x86-64-avx10-vpmldsabflydq.s

.*: +file format .*

Disassembly of section \.text:

#...
[a-f0-9]+ <_intel>:
[ 	]*[a-f0-9]+:[ 	]*62 03 95 00 dc f4 7b[ 	]*vpmldsabflydq xmm30,xmm29,xmm28,0x7b
[ 	]*[a-f0-9]+:[ 	]*62 03 95 20 dc f4 7b[ 	]*vpmldsabflydq ymm30,ymm29,ymm28,0x7b
[ 	]*[a-f0-9]+:[ 	]*62 03 95 40 dc f4 7b[ 	]*vpmldsabflydq zmm30,zmm29,zmm28,0x7b
[ 	]*[a-f0-9]+:[ 	]*62 23 95 00 dc b4 f5 00 00 00 10 7b[ 	]*vpmldsabflydq xmm30,xmm29,XMMWORD PTR \[rbp\+r14\*8\+0x10000000\],0x7b
[ 	]*[a-f0-9]+:[ 	]*62 43 95 00 dc 31 7b[ 	]*vpmldsabflydq xmm30,xmm29,XMMWORD PTR \[r9\],0x7b
[ 	]*[a-f0-9]+:[ 	]*62 63 95 00 dc 71 7f 7b[ 	]*vpmldsabflydq xmm30,xmm29,XMMWORD PTR \[rcx\+0x7f0\],0x7b
[ 	]*[a-f0-9]+:[ 	]*62 63 95 00 dc 72 80 7b[ 	]*vpmldsabflydq xmm30,xmm29,XMMWORD PTR \[rdx-0x800\],0x7b
[ 	]*[a-f0-9]+:[ 	]*62 23 95 20 dc b4 f5 00 00 00 10 7b[ 	]*vpmldsabflydq ymm30,ymm29,YMMWORD PTR \[rbp\+r14\*8\+0x10000000\],0x7b
[ 	]*[a-f0-9]+:[ 	]*62 43 95 20 dc 31 7b[ 	]*vpmldsabflydq ymm30,ymm29,YMMWORD PTR \[r9\],0x7b
[ 	]*[a-f0-9]+:[ 	]*62 63 95 20 dc 71 7f 7b[ 	]*vpmldsabflydq ymm30,ymm29,YMMWORD PTR \[rcx\+0xfe0\],0x7b
[ 	]*[a-f0-9]+:[ 	]*62 63 95 20 dc 72 80 7b[ 	]*vpmldsabflydq ymm30,ymm29,YMMWORD PTR \[rdx-0x1000\],0x7b
[ 	]*[a-f0-9]+:[ 	]*62 23 95 40 dc b4 f5 00 00 00 10 7b[ 	]*vpmldsabflydq zmm30,zmm29,ZMMWORD PTR \[rbp\+r14\*8\+0x10000000\],0x7b
[ 	]*[a-f0-9]+:[ 	]*62 43 95 40 dc 31 7b[ 	]*vpmldsabflydq zmm30,zmm29,ZMMWORD PTR \[r9\],0x7b
[ 	]*[a-f0-9]+:[ 	]*62 63 95 40 dc 71 7f 7b[ 	]*vpmldsabflydq zmm30,zmm29,ZMMWORD PTR \[rcx\+0x1fc0\],0x7b
[ 	]*[a-f0-9]+:[ 	]*62 63 95 40 dc 72 80 7b[ 	]*vpmldsabflydq zmm30,zmm29,ZMMWORD PTR \[rdx-0x2000\],0x7b
#pass
