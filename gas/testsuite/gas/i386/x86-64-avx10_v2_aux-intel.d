#objdump: -dw -Mintel
#name: x86-64 AVX10 V2 AUX (Intel disassembly)
#source: x86-64-avx10_v2_aux.s

.*: +file format .*

Disassembly of section \.text:

#...
[a-f0-9]+ <_intel>:
[ 	]*[a-f0-9]+:[ 	]*62 f5 7e 08 39 c1[ 	]+vcvtps2bf8[ 	]+xmm0,xmm1
[ 	]*[a-f0-9]+:[ 	]*62 f5 7e 28 39 c1[ 	]+vcvtps2bf8[ 	]+xmm0,ymm1
[ 	]*[a-f0-9]+:[ 	]*62 f5 7e 48 39 c1[ 	]+vcvtps2bf8[ 	]+xmm0,zmm1
[ 	]*[a-f0-9]+:[ 	]*62 f5 7e 08 39 41 7f[ 	]+vcvtps2bf8[ 	]+xmm0,XMMWORD PTR \[rcx\+0x7f0\]
[ 	]*[a-f0-9]+:[ 	]*62 f5 7e 28 39 41 7f[ 	]+vcvtps2bf8[ 	]+xmm0,YMMWORD PTR \[rcx\+0xfe0\]
[ 	]*[a-f0-9]+:[ 	]*62 f5 7e 48 39 41 7f[ 	]+vcvtps2bf8[ 	]+xmm0,ZMMWORD PTR \[rcx\+0x1fc0\]
[ 	]*[a-f0-9]+:[ 	]*62 d5 7e 18 39 01[ 	]+vcvtps2bf8[ 	]+xmm0,DWORD BCST \[r9\]\{1to4\}
[ 	]*[a-f0-9]+:[ 	]*62 d5 7e 38 39 01[ 	]+vcvtps2bf8[ 	]+xmm0,DWORD BCST \[r9\]\{1to8\}
[ 	]*[a-f0-9]+:[ 	]*62 d5 7e 58 39 01[ 	]+vcvtps2bf8[ 	]+xmm0,DWORD BCST \[r9\]\{1to16\}
[ 	]*[a-f0-9]+:[ 	]*62 f5 7e 09 39 c1[ 	]+vcvtps2bf8[ 	]+xmm0\{k1\},xmm1
[ 	]*[a-f0-9]+:[ 	]*62 f5 7e 89 39 c1[ 	]+vcvtps2bf8[ 	]+xmm0\{k1\}\{z\},xmm1
[ 	]*[a-f0-9]+:[ 	]*62 f5 74 08 39 c2[ 	]+vcvtbiasps2bf8[ 	]+xmm0,xmm1,xmm2
[ 	]*[a-f0-9]+:[ 	]*62 f5 74 28 39 c2[ 	]+vcvtbiasps2bf8[ 	]+xmm0,ymm1,ymm2
[ 	]*[a-f0-9]+:[ 	]*62 f5 74 48 39 c2[ 	]+vcvtbiasps2bf8[ 	]+xmm0,zmm1,zmm2
[ 	]*[a-f0-9]+:[ 	]*62 f5 74 08 39 41 7f[ 	]+vcvtbiasps2bf8[ 	]+xmm0,xmm1,XMMWORD PTR \[rcx\+0x7f0\]
[ 	]*[a-f0-9]+:[ 	]*62 f5 74 28 39 41 7f[ 	]+vcvtbiasps2bf8[ 	]+xmm0,ymm1,YMMWORD PTR \[rcx\+0xfe0\]
[ 	]*[a-f0-9]+:[ 	]*62 f5 74 48 39 41 7f[ 	]+vcvtbiasps2bf8[ 	]+xmm0,zmm1,ZMMWORD PTR \[rcx\+0x1fc0\]
[ 	]*[a-f0-9]+:[ 	]*62 d5 74 18 39 01[ 	]+vcvtbiasps2bf8[ 	]+xmm0,xmm1,DWORD BCST \[r9\]
[ 	]*[a-f0-9]+:[ 	]*62 d5 74 38 39 01[ 	]+vcvtbiasps2bf8[ 	]+xmm0,ymm1,DWORD BCST \[r9\]
[ 	]*[a-f0-9]+:[ 	]*62 d5 74 58 39 01[ 	]+vcvtbiasps2bf8[ 	]+xmm0,zmm1,DWORD BCST \[r9\]
[ 	]*[a-f0-9]+:[ 	]*62 f5 74 09 39 c2[ 	]+vcvtbiasps2bf8[ 	]+xmm0\{k1\},xmm1,xmm2
[ 	]*[a-f0-9]+:[ 	]*62 f5 74 89 39 c2[ 	]+vcvtbiasps2bf8[ 	]+xmm0\{k1\}\{z\},xmm1,xmm2
[ 	]*[a-f0-9]+:[ 	]*62 f5 7e 08 3b c1[ 	]+vcvtps2bf8s[ 	]+xmm0,xmm1
[ 	]*[a-f0-9]+:[ 	]*62 f5 7e 28 3b c1[ 	]+vcvtps2bf8s[ 	]+xmm0,ymm1
[ 	]*[a-f0-9]+:[ 	]*62 f5 7e 48 3b c1[ 	]+vcvtps2bf8s[ 	]+xmm0,zmm1
[ 	]*[a-f0-9]+:[ 	]*62 f5 7e 08 3b 41 7f[ 	]+vcvtps2bf8s[ 	]+xmm0,XMMWORD PTR \[rcx\+0x7f0\]
[ 	]*[a-f0-9]+:[ 	]*62 f5 7e 28 3b 41 7f[ 	]+vcvtps2bf8s[ 	]+xmm0,YMMWORD PTR \[rcx\+0xfe0\]
[ 	]*[a-f0-9]+:[ 	]*62 f5 7e 48 3b 41 7f[ 	]+vcvtps2bf8s[ 	]+xmm0,ZMMWORD PTR \[rcx\+0x1fc0\]
[ 	]*[a-f0-9]+:[ 	]*62 d5 7e 18 3b 01[ 	]+vcvtps2bf8s[ 	]+xmm0,DWORD BCST \[r9\]\{1to4\}
[ 	]*[a-f0-9]+:[ 	]*62 d5 7e 38 3b 01[ 	]+vcvtps2bf8s[ 	]+xmm0,DWORD BCST \[r9\]\{1to8\}
[ 	]*[a-f0-9]+:[ 	]*62 d5 7e 58 3b 01[ 	]+vcvtps2bf8s[ 	]+xmm0,DWORD BCST \[r9\]\{1to16\}
[ 	]*[a-f0-9]+:[ 	]*62 f5 7e 09 3b c1[ 	]+vcvtps2bf8s[ 	]+xmm0\{k1\},xmm1
[ 	]*[a-f0-9]+:[ 	]*62 f5 7e 89 3b c1[ 	]+vcvtps2bf8s[ 	]+xmm0\{k1\}\{z\},xmm1
[ 	]*[a-f0-9]+:[ 	]*62 f5 74 08 3b c2[ 	]+vcvtbiasps2bf8s[ 	]+xmm0,xmm1,xmm2
[ 	]*[a-f0-9]+:[ 	]*62 f5 74 28 3b c2[ 	]+vcvtbiasps2bf8s[ 	]+xmm0,ymm1,ymm2
[ 	]*[a-f0-9]+:[ 	]*62 f5 74 48 3b c2[ 	]+vcvtbiasps2bf8s[ 	]+xmm0,zmm1,zmm2
[ 	]*[a-f0-9]+:[ 	]*62 f5 74 08 3b 41 7f[ 	]+vcvtbiasps2bf8s[ 	]+xmm0,xmm1,XMMWORD PTR \[rcx\+0x7f0\]
[ 	]*[a-f0-9]+:[ 	]*62 f5 74 28 3b 41 7f[ 	]+vcvtbiasps2bf8s[ 	]+xmm0,ymm1,YMMWORD PTR \[rcx\+0xfe0\]
[ 	]*[a-f0-9]+:[ 	]*62 f5 74 48 3b 41 7f[ 	]+vcvtbiasps2bf8s[ 	]+xmm0,zmm1,ZMMWORD PTR \[rcx\+0x1fc0\]
[ 	]*[a-f0-9]+:[ 	]*62 d5 74 18 3b 01[ 	]+vcvtbiasps2bf8s[ 	]+xmm0,xmm1,DWORD BCST \[r9\]
[ 	]*[a-f0-9]+:[ 	]*62 d5 74 38 3b 01[ 	]+vcvtbiasps2bf8s[ 	]+xmm0,ymm1,DWORD BCST \[r9\]
[ 	]*[a-f0-9]+:[ 	]*62 d5 74 58 3b 01[ 	]+vcvtbiasps2bf8s[ 	]+xmm0,zmm1,DWORD BCST \[r9\]
[ 	]*[a-f0-9]+:[ 	]*62 f5 74 09 3b c2[ 	]+vcvtbiasps2bf8s[ 	]+xmm0\{k1\},xmm1,xmm2
[ 	]*[a-f0-9]+:[ 	]*62 f5 74 89 3b c2[ 	]+vcvtbiasps2bf8s[ 	]+xmm0\{k1\}\{z\},xmm1,xmm2
[ 	]*[a-f0-9]+:[ 	]*62 f5 7e 08 38 c1[ 	]+vcvtps2hf8[ 	]+xmm0,xmm1
[ 	]*[a-f0-9]+:[ 	]*62 f5 7e 28 38 c1[ 	]+vcvtps2hf8[ 	]+xmm0,ymm1
[ 	]*[a-f0-9]+:[ 	]*62 f5 7e 48 38 c1[ 	]+vcvtps2hf8[ 	]+xmm0,zmm1
[ 	]*[a-f0-9]+:[ 	]*62 f5 7e 08 38 41 7f[ 	]+vcvtps2hf8[ 	]+xmm0,XMMWORD PTR \[rcx\+0x7f0\]
[ 	]*[a-f0-9]+:[ 	]*62 f5 7e 28 38 41 7f[ 	]+vcvtps2hf8[ 	]+xmm0,YMMWORD PTR \[rcx\+0xfe0\]
[ 	]*[a-f0-9]+:[ 	]*62 f5 7e 48 38 41 7f[ 	]+vcvtps2hf8[ 	]+xmm0,ZMMWORD PTR \[rcx\+0x1fc0\]
[ 	]*[a-f0-9]+:[ 	]*62 d5 7e 18 38 01[ 	]+vcvtps2hf8[ 	]+xmm0,DWORD BCST \[r9\]\{1to4\}
[ 	]*[a-f0-9]+:[ 	]*62 d5 7e 38 38 01[ 	]+vcvtps2hf8[ 	]+xmm0,DWORD BCST \[r9\]\{1to8\}
[ 	]*[a-f0-9]+:[ 	]*62 d5 7e 58 38 01[ 	]+vcvtps2hf8[ 	]+xmm0,DWORD BCST \[r9\]\{1to16\}
[ 	]*[a-f0-9]+:[ 	]*62 f5 7e 09 38 c1[ 	]+vcvtps2hf8[ 	]+xmm0\{k1\},xmm1
[ 	]*[a-f0-9]+:[ 	]*62 f5 7e 89 38 c1[ 	]+vcvtps2hf8[ 	]+xmm0\{k1\}\{z\},xmm1
[ 	]*[a-f0-9]+:[ 	]*62 f5 74 08 38 c2[ 	]+vcvtbiasps2hf8[ 	]+xmm0,xmm1,xmm2
[ 	]*[a-f0-9]+:[ 	]*62 f5 74 28 38 c2[ 	]+vcvtbiasps2hf8[ 	]+xmm0,ymm1,ymm2
[ 	]*[a-f0-9]+:[ 	]*62 f5 74 48 38 c2[ 	]+vcvtbiasps2hf8[ 	]+xmm0,zmm1,zmm2
[ 	]*[a-f0-9]+:[ 	]*62 f5 74 08 38 41 7f[ 	]+vcvtbiasps2hf8[ 	]+xmm0,xmm1,XMMWORD PTR \[rcx\+0x7f0\]
[ 	]*[a-f0-9]+:[ 	]*62 f5 74 28 38 41 7f[ 	]+vcvtbiasps2hf8[ 	]+xmm0,ymm1,YMMWORD PTR \[rcx\+0xfe0\]
[ 	]*[a-f0-9]+:[ 	]*62 f5 74 48 38 41 7f[ 	]+vcvtbiasps2hf8[ 	]+xmm0,zmm1,ZMMWORD PTR \[rcx\+0x1fc0\]
[ 	]*[a-f0-9]+:[ 	]*62 d5 74 18 38 01[ 	]+vcvtbiasps2hf8[ 	]+xmm0,xmm1,DWORD BCST \[r9\]
[ 	]*[a-f0-9]+:[ 	]*62 d5 74 38 38 01[ 	]+vcvtbiasps2hf8[ 	]+xmm0,ymm1,DWORD BCST \[r9\]
[ 	]*[a-f0-9]+:[ 	]*62 d5 74 58 38 01[ 	]+vcvtbiasps2hf8[ 	]+xmm0,zmm1,DWORD BCST \[r9\]
[ 	]*[a-f0-9]+:[ 	]*62 f5 74 09 38 c2[ 	]+vcvtbiasps2hf8[ 	]+xmm0\{k1\},xmm1,xmm2
[ 	]*[a-f0-9]+:[ 	]*62 f5 74 89 38 c2[ 	]+vcvtbiasps2hf8[ 	]+xmm0\{k1\}\{z\},xmm1,xmm2
[ 	]*[a-f0-9]+:[ 	]*62 f5 7e 08 3a c1[ 	]+vcvtps2hf8s[ 	]+xmm0,xmm1
[ 	]*[a-f0-9]+:[ 	]*62 f5 7e 28 3a c1[ 	]+vcvtps2hf8s[ 	]+xmm0,ymm1
[ 	]*[a-f0-9]+:[ 	]*62 f5 7e 48 3a c1[ 	]+vcvtps2hf8s[ 	]+xmm0,zmm1
[ 	]*[a-f0-9]+:[ 	]*62 f5 7e 08 3a 41 7f[ 	]+vcvtps2hf8s[ 	]+xmm0,XMMWORD PTR \[rcx\+0x7f0\]
[ 	]*[a-f0-9]+:[ 	]*62 f5 7e 28 3a 41 7f[ 	]+vcvtps2hf8s[ 	]+xmm0,YMMWORD PTR \[rcx\+0xfe0\]
[ 	]*[a-f0-9]+:[ 	]*62 f5 7e 48 3a 41 7f[ 	]+vcvtps2hf8s[ 	]+xmm0,ZMMWORD PTR \[rcx\+0x1fc0\]
[ 	]*[a-f0-9]+:[ 	]*62 d5 7e 18 3a 01[ 	]+vcvtps2hf8s[ 	]+xmm0,DWORD BCST \[r9\]\{1to4\}
[ 	]*[a-f0-9]+:[ 	]*62 d5 7e 38 3a 01[ 	]+vcvtps2hf8s[ 	]+xmm0,DWORD BCST \[r9\]\{1to8\}
[ 	]*[a-f0-9]+:[ 	]*62 d5 7e 58 3a 01[ 	]+vcvtps2hf8s[ 	]+xmm0,DWORD BCST \[r9\]\{1to16\}
[ 	]*[a-f0-9]+:[ 	]*62 f5 7e 09 3a c1[ 	]+vcvtps2hf8s[ 	]+xmm0\{k1\},xmm1
[ 	]*[a-f0-9]+:[ 	]*62 f5 7e 89 3a c1[ 	]+vcvtps2hf8s[ 	]+xmm0\{k1\}\{z\},xmm1
[ 	]*[a-f0-9]+:[ 	]*62 f5 74 08 3a c2[ 	]+vcvtbiasps2hf8s[ 	]+xmm0,xmm1,xmm2
[ 	]*[a-f0-9]+:[ 	]*62 f5 74 28 3a c2[ 	]+vcvtbiasps2hf8s[ 	]+xmm0,ymm1,ymm2
[ 	]*[a-f0-9]+:[ 	]*62 f5 74 48 3a c2[ 	]+vcvtbiasps2hf8s[ 	]+xmm0,zmm1,zmm2
[ 	]*[a-f0-9]+:[ 	]*62 f5 74 08 3a 41 7f[ 	]+vcvtbiasps2hf8s[ 	]+xmm0,xmm1,XMMWORD PTR \[rcx\+0x7f0\]
[ 	]*[a-f0-9]+:[ 	]*62 f5 74 28 3a 41 7f[ 	]+vcvtbiasps2hf8s[ 	]+xmm0,ymm1,YMMWORD PTR \[rcx\+0xfe0\]
[ 	]*[a-f0-9]+:[ 	]*62 f5 74 48 3a 41 7f[ 	]+vcvtbiasps2hf8s[ 	]+xmm0,zmm1,ZMMWORD PTR \[rcx\+0x1fc0\]
[ 	]*[a-f0-9]+:[ 	]*62 d5 74 18 3a 01[ 	]+vcvtbiasps2hf8s[ 	]+xmm0,xmm1,DWORD BCST \[r9\]
[ 	]*[a-f0-9]+:[ 	]*62 d5 74 38 3a 01[ 	]+vcvtbiasps2hf8s[ 	]+xmm0,ymm1,DWORD BCST \[r9\]
[ 	]*[a-f0-9]+:[ 	]*62 d5 74 58 3a 01[ 	]+vcvtbiasps2hf8s[ 	]+xmm0,zmm1,DWORD BCST \[r9\]
[ 	]*[a-f0-9]+:[ 	]*62 f5 74 09 3a c2[ 	]+vcvtbiasps2hf8s[ 	]+xmm0\{k1\},xmm1,xmm2
[ 	]*[a-f0-9]+:[ 	]*62 f5 74 89 3a c2[ 	]+vcvtbiasps2hf8s[ 	]+xmm0\{k1\}\{z\},xmm1,xmm2
[ 	]*[a-f0-9]+:[ 	]*62 f5 7d 08 38 c1[ 	]+vcvtrops2hf8[ 	]+xmm0,xmm1
[ 	]*[a-f0-9]+:[ 	]*62 f5 7d 28 38 c1[ 	]+vcvtrops2hf8[ 	]+xmm0,ymm1
[ 	]*[a-f0-9]+:[ 	]*62 f5 7d 48 38 c1[ 	]+vcvtrops2hf8[ 	]+xmm0,zmm1
[ 	]*[a-f0-9]+:[ 	]*62 f5 7d 08 38 41 7f[ 	]+vcvtrops2hf8[ 	]+xmm0,XMMWORD PTR \[rcx\+0x7f0\]
[ 	]*[a-f0-9]+:[ 	]*62 f5 7d 28 38 41 7f[ 	]+vcvtrops2hf8[ 	]+xmm0,YMMWORD PTR \[rcx\+0xfe0\]
[ 	]*[a-f0-9]+:[ 	]*62 f5 7d 48 38 41 7f[ 	]+vcvtrops2hf8[ 	]+xmm0,ZMMWORD PTR \[rcx\+0x1fc0\]
[ 	]*[a-f0-9]+:[ 	]*62 d5 7d 18 38 01[ 	]+vcvtrops2hf8[ 	]+xmm0,DWORD BCST \[r9\]\{1to4\}
[ 	]*[a-f0-9]+:[ 	]*62 d5 7d 38 38 01[ 	]+vcvtrops2hf8[ 	]+xmm0,DWORD BCST \[r9\]\{1to8\}
[ 	]*[a-f0-9]+:[ 	]*62 d5 7d 58 38 01[ 	]+vcvtrops2hf8[ 	]+xmm0,DWORD BCST \[r9\]\{1to16\}
[ 	]*[a-f0-9]+:[ 	]*62 f5 7d 09 38 c1[ 	]+vcvtrops2hf8[ 	]+xmm0\{k1\},xmm1
[ 	]*[a-f0-9]+:[ 	]*62 f5 7d 89 38 c1[ 	]+vcvtrops2hf8[ 	]+xmm0\{k1\}\{z\},xmm1
[ 	]*[a-f0-9]+:[ 	]*62 f5 7d 08 3a c1[ 	]+vcvtrops2hf8s[ 	]+xmm0,xmm1
[ 	]*[a-f0-9]+:[ 	]*62 f5 7d 28 3a c1[ 	]+vcvtrops2hf8s[ 	]+xmm0,ymm1
[ 	]*[a-f0-9]+:[ 	]*62 f5 7d 48 3a c1[ 	]+vcvtrops2hf8s[ 	]+xmm0,zmm1
[ 	]*[a-f0-9]+:[ 	]*62 f5 7d 08 3a 41 7f[ 	]+vcvtrops2hf8s[ 	]+xmm0,XMMWORD PTR \[rcx\+0x7f0\]
[ 	]*[a-f0-9]+:[ 	]*62 f5 7d 28 3a 41 7f[ 	]+vcvtrops2hf8s[ 	]+xmm0,YMMWORD PTR \[rcx\+0xfe0\]
[ 	]*[a-f0-9]+:[ 	]*62 f5 7d 48 3a 41 7f[ 	]+vcvtrops2hf8s[ 	]+xmm0,ZMMWORD PTR \[rcx\+0x1fc0\]
[ 	]*[a-f0-9]+:[ 	]*62 d5 7d 18 3a 01[ 	]+vcvtrops2hf8s[ 	]+xmm0,DWORD BCST \[r9\]\{1to4\}
[ 	]*[a-f0-9]+:[ 	]*62 d5 7d 38 3a 01[ 	]+vcvtrops2hf8s[ 	]+xmm0,DWORD BCST \[r9\]\{1to8\}
[ 	]*[a-f0-9]+:[ 	]*62 d5 7d 58 3a 01[ 	]+vcvtrops2hf8s[ 	]+xmm0,DWORD BCST \[r9\]\{1to16\}
[ 	]*[a-f0-9]+:[ 	]*62 f5 7d 09 3a c1[ 	]+vcvtrops2hf8s[ 	]+xmm0\{k1\},xmm1
[ 	]*[a-f0-9]+:[ 	]*62 f5 7d 89 3a c1[ 	]+vcvtrops2hf8s[ 	]+xmm0\{k1\}\{z\},xmm1
[ 	]*[a-f0-9]+:[ 	]*62 f5 fc 08 36 c1[ 	]+vcvtbf82ps[ 	]+xmm0,xmm1
[ 	]*[a-f0-9]+:[ 	]*62 f5 fc 28 36 c1[ 	]+vcvtbf82ps[ 	]+ymm0,xmm1
[ 	]*[a-f0-9]+:[ 	]*62 f5 fc 48 36 c1[ 	]+vcvtbf82ps[ 	]+zmm0,xmm1
[ 	]*[a-f0-9]+:[ 	]*62 f5 fc 08 36 41 7f[ 	]+vcvtbf82ps[ 	]+xmm0,DWORD PTR \[rcx\+0x1fc\]
[ 	]*[a-f0-9]+:[ 	]*62 f5 fc 28 36 41 7f[ 	]+vcvtbf82ps[ 	]+ymm0,QWORD PTR \[rcx\+0x3f8\]
[ 	]*[a-f0-9]+:[ 	]*62 f5 fc 48 36 41 7f[ 	]+vcvtbf82ps[ 	]+zmm0,XMMWORD PTR \[rcx\+0x7f0\]
[ 	]*[a-f0-9]+:[ 	]*62 f5 fc 09 36 c1[ 	]+vcvtbf82ps[ 	]+xmm0\{k1\},xmm1
[ 	]*[a-f0-9]+:[ 	]*62 f5 fc 89 36 c1[ 	]+vcvtbf82ps[ 	]+xmm0\{k1\}\{z\},xmm1
[ 	]*[a-f0-9]+:[ 	]*62 f5 fe 08 3d c8[ 	]+vcvtbf82bf4s[ 	]+xmm0,xmm1
[ 	]*[a-f0-9]+:[ 	]*62 f5 fe 28 3d c8[ 	]+vcvtbf82bf4s[ 	]+xmm0,ymm1
[ 	]*[a-f0-9]+:[ 	]*62 f5 fe 48 3d c8[ 	]+vcvtbf82bf4s[ 	]+ymm0,zmm1
[ 	]*[a-f0-9]+:[ 	]*62 f5 fe 08 3d 49 7f[ 	]+vcvtbf82bf4s[ 	]+QWORD PTR \[rcx\+0x3f8\],xmm1
[ 	]*[a-f0-9]+:[ 	]*62 f5 fe 28 3d 49 7f[ 	]+vcvtbf82bf4s[ 	]+XMMWORD PTR \[rcx\+0x7f0\],ymm1
[ 	]*[a-f0-9]+:[ 	]*62 f5 fe 48 3d 49 7f[ 	]+vcvtbf82bf4s[ 	]+YMMWORD PTR \[rcx\+0xfe0\],zmm1
[ 	]*[a-f0-9]+:[ 	]*62 f5 7c 08 36 c1[ 	]+vcvthf82ps[ 	]+xmm0,xmm1
[ 	]*[a-f0-9]+:[ 	]*62 f5 7c 28 36 c1[ 	]+vcvthf82ps[ 	]+ymm0,xmm1
[ 	]*[a-f0-9]+:[ 	]*62 f5 7c 48 36 c1[ 	]+vcvthf82ps[ 	]+zmm0,xmm1
[ 	]*[a-f0-9]+:[ 	]*62 f5 7c 08 36 41 7f[ 	]+vcvthf82ps[ 	]+xmm0,DWORD PTR \[rcx\+0x1fc\]
[ 	]*[a-f0-9]+:[ 	]*62 f5 7c 28 36 41 7f[ 	]+vcvthf82ps[ 	]+ymm0,QWORD PTR \[rcx\+0x3f8\]
[ 	]*[a-f0-9]+:[ 	]*62 f5 7c 48 36 41 7f[ 	]+vcvthf82ps[ 	]+zmm0,XMMWORD PTR \[rcx\+0x7f0\]
[ 	]*[a-f0-9]+:[ 	]*62 f5 7c 09 36 c1[ 	]+vcvthf82ps[ 	]+xmm0\{k1\},xmm1
[ 	]*[a-f0-9]+:[ 	]*62 f5 7c 89 36 c1[ 	]+vcvthf82ps[ 	]+xmm0\{k1\}\{z\},xmm1
[ 	]*[a-f0-9]+:[ 	]*62 f5 7e 08 3d c8[ 	]+vcvthf82bf4s[ 	]+xmm0,xmm1
[ 	]*[a-f0-9]+:[ 	]*62 f5 7e 28 3d c8[ 	]+vcvthf82bf4s[ 	]+xmm0,ymm1
[ 	]*[a-f0-9]+:[ 	]*62 f5 7e 48 3d c8[ 	]+vcvthf82bf4s[ 	]+ymm0,zmm1
[ 	]*[a-f0-9]+:[ 	]*62 f5 7e 08 3d 49 7f[ 	]+vcvthf82bf4s[ 	]+QWORD PTR \[rcx\+0x3f8\],xmm1
[ 	]*[a-f0-9]+:[ 	]*62 f5 7e 28 3d 49 7f[ 	]+vcvthf82bf4s[ 	]+XMMWORD PTR \[rcx\+0x7f0\],ymm1
[ 	]*[a-f0-9]+:[ 	]*62 f5 7e 48 3d 49 7f[ 	]+vcvthf82bf4s[ 	]+YMMWORD PTR \[rcx\+0xfe0\],zmm1
[ 	]*[a-f0-9]+:[ 	]*62 f5 fe 08 3e c1[ 	]+vcvtbf82bf6s[ 	]+xmm0,xmm1
[ 	]*[a-f0-9]+:[ 	]*62 f5 fe 28 3e c1[ 	]+vcvtbf82bf6s[ 	]+ymm0,ymm1
[ 	]*[a-f0-9]+:[ 	]*62 f5 fe 48 3e c1[ 	]+vcvtbf82bf6s[ 	]+zmm0,zmm1
[ 	]*[a-f0-9]+:[ 	]*62 f5 7e 08 3c c1[ 	]+vcvthf82hf6s[ 	]+xmm0,xmm1
[ 	]*[a-f0-9]+:[ 	]*62 f5 7e 28 3c c1[ 	]+vcvthf82hf6s[ 	]+ymm0,ymm1
[ 	]*[a-f0-9]+:[ 	]*62 f5 7e 48 3c c1[ 	]+vcvthf82hf6s[ 	]+zmm0,zmm1
[ 	]*[a-f0-9]+:[ 	]*62 f5 7c 08 37 c1[ 	]+vcvtbf42hf8[ 	]+xmm0,xmm1
[ 	]*[a-f0-9]+:[ 	]*62 f5 7c 28 37 c1[ 	]+vcvtbf42hf8[ 	]+ymm0,xmm1
[ 	]*[a-f0-9]+:[ 	]*62 f5 7c 48 37 c1[ 	]+vcvtbf42hf8[ 	]+zmm0,ymm1
[ 	]*[a-f0-9]+:[ 	]*62 f5 7c 08 37 41 7f[ 	]+vcvtbf42hf8[ 	]+xmm0,QWORD PTR \[rcx\+0x3f8\]
[ 	]*[a-f0-9]+:[ 	]*62 f5 7c 28 37 41 7f[ 	]+vcvtbf42hf8[ 	]+ymm0,XMMWORD PTR \[rcx\+0x7f0\]
[ 	]*[a-f0-9]+:[ 	]*62 f5 7c 48 37 41 7f[ 	]+vcvtbf42hf8[ 	]+zmm0,YMMWORD PTR \[rcx\+0xfe0\]
[ 	]*[a-f0-9]+:[ 	]*62 f5 7c 09 37 c1[ 	]+vcvtbf42hf8[ 	]+xmm0\{k1\},xmm1
[ 	]*[a-f0-9]+:[ 	]*62 f5 7c 89 37 c1[ 	]+vcvtbf42hf8[ 	]+xmm0\{k1\}\{z\},xmm1
[ 	]*[a-f0-9]+:[ 	]*62 f5 fd 08 37 c1[ 	]+vcvtbf62hf8[ 	]+xmm0,xmm1
[ 	]*[a-f0-9]+:[ 	]*62 f5 fd 28 37 c1[ 	]+vcvtbf62hf8[ 	]+ymm0,ymm1
[ 	]*[a-f0-9]+:[ 	]*62 f5 fd 48 37 c1[ 	]+vcvtbf62hf8[ 	]+zmm0,zmm1
[ 	]*[a-f0-9]+:[ 	]*62 f5 fd 09 37 c1[ 	]+vcvtbf62hf8[ 	]+xmm0\{k1\},xmm1
[ 	]*[a-f0-9]+:[ 	]*62 f5 fd 89 37 c1[ 	]+vcvtbf62hf8[ 	]+xmm0\{k1\}\{z\},xmm1
[ 	]*[a-f0-9]+:[ 	]*62 f5 7d 08 37 c1[ 	]+vcvthf62hf8[ 	]+xmm0,xmm1
[ 	]*[a-f0-9]+:[ 	]*62 f5 7d 28 37 c1[ 	]+vcvthf62hf8[ 	]+ymm0,ymm1
[ 	]*[a-f0-9]+:[ 	]*62 f5 7d 48 37 c1[ 	]+vcvthf62hf8[ 	]+zmm0,zmm1
[ 	]*[a-f0-9]+:[ 	]*62 f5 7d 09 37 c1[ 	]+vcvthf62hf8[ 	]+xmm0\{k1\},xmm1
[ 	]*[a-f0-9]+:[ 	]*62 f5 7d 89 37 c1[ 	]+vcvthf62hf8[ 	]+xmm0\{k1\}\{z\},xmm1
[ 	]*[a-f0-9]+:[ 	]*62 f2 7e 08 41 c8[ 	]+vpmovssdb[ 	]+xmm0,xmm1
[ 	]*[a-f0-9]+:[ 	]*62 f2 7e 28 41 c8[ 	]+vpmovssdb[ 	]+xmm0,ymm1
[ 	]*[a-f0-9]+:[ 	]*62 f2 7e 48 41 c8[ 	]+vpmovssdb[ 	]+xmm0,zmm1
[ 	]*[a-f0-9]+:[ 	]*62 f2 7e 08 41 49 7f[ 	]+vpmovssdb[ 	]+DWORD PTR \[rcx\+0x1fc\],xmm1
[ 	]*[a-f0-9]+:[ 	]*62 f2 7e 28 41 49 7f[ 	]+vpmovssdb[ 	]+QWORD PTR \[rcx\+0x3f8\],ymm1
[ 	]*[a-f0-9]+:[ 	]*62 f2 7e 48 41 49 7f[ 	]+vpmovssdb[ 	]+XMMWORD PTR \[rcx\+0x7f0\],zmm1
[ 	]*[a-f0-9]+:[ 	]*62 f2 7e 09 41 c8[ 	]+vpmovssdb[ 	]+xmm0\{k1\},xmm1
[ 	]*[a-f0-9]+:[ 	]*62 f2 7e 89 41 c8[ 	]+vpmovssdb[ 	]+xmm0\{k1\}\{z\},xmm1
[ 	]*[a-f0-9]+:[ 	]*62 f3 7c 08 3d c1 10[ 	]+vunpackb[ 	]+xmm0,xmm1,0x10
[ 	]*[a-f0-9]+:[ 	]*62 f3 7c 28 3d c1 10[ 	]+vunpackb[ 	]+ymm0,ymm1,0x10
[ 	]*[a-f0-9]+:[ 	]*62 f3 7c 48 3d c1 10[ 	]+vunpackb[ 	]+zmm0,zmm1,0x10
[ 	]*[a-f0-9]+:[ 	]*62 f3 7c 08 3d 41 7f 10[ 	]+vunpackb[ 	]+xmm0,XMMWORD PTR \[rcx\+0x7f0\],0x10
[ 	]*[a-f0-9]+:[ 	]*62 f3 7c 28 3d 41 7f 10[ 	]+vunpackb[ 	]+ymm0,YMMWORD PTR \[rcx\+0xfe0\],0x10
[ 	]*[a-f0-9]+:[ 	]*62 f3 7c 48 3d 41 7f 10[ 	]+vunpackb[ 	]+zmm0,ZMMWORD PTR \[rcx\+0x1fc0\],0x10
[ 	]*[a-f0-9]+:[ 	]*62 f3 7c 09 3d c1 10[ 	]+vunpackb[ 	]+xmm0\{k1\},xmm1,0x10
[ 	]*[a-f0-9]+:[ 	]*62 f3 7c 89 3d c1 10[ 	]+vunpackb[ 	]+xmm0\{k1\}\{z\},xmm1,0x10
#pass
