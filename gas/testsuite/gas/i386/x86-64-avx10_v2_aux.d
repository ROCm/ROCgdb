#objdump: -dw -Mx86-64
#name: x86-64 AVX10 V2 AUX

.*: +file format .*

Disassembly of section \.text:

0+ <_start>:
[ 	]*[a-f0-9]+:[ 	]*62 f5 7e 08 39 c1[ 	]+vcvtps2bf8[ 	]+%xmm1,%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 f5 7e 28 39 c1[ 	]+vcvtps2bf8[ 	]+%ymm1,%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 f5 7e 48 39 c1[ 	]+vcvtps2bf8[ 	]+%zmm1,%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 f5 7e 08 39 41 7f[ 	]+vcvtps2bf8x[ 	]+0x7f0\(%rcx\),%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 f5 7e 28 39 41 7f[ 	]+vcvtps2bf8y[ 	]+0xfe0\(%rcx\),%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 f5 7e 48 39 41 7f[ 	]+vcvtps2bf8z[ 	]+0x1fc0\(%rcx\),%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 d5 7e 18 39 01[ 	]+vcvtps2bf8[ 	]+\(%r9\)\{1to4\},%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 d5 7e 38 39 01[ 	]+vcvtps2bf8[ 	]+\(%r9\)\{1to8\},%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 d5 7e 58 39 01[ 	]+vcvtps2bf8[ 	]+\(%r9\)\{1to16\},%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 f5 7e 09 39 c1[ 	]+vcvtps2bf8[ 	]+%xmm1,%xmm0\{%k1\}
[ 	]*[a-f0-9]+:[ 	]*62 f5 7e 89 39 c1[ 	]+vcvtps2bf8[ 	]+%xmm1,%xmm0\{%k1\}\{z\}
[ 	]*[a-f0-9]+:[ 	]*62 f5 74 08 39 c2[ 	]+vcvtbiasps2bf8[ 	]+%xmm2,%xmm1,%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 f5 74 28 39 c2[ 	]+vcvtbiasps2bf8[ 	]+%ymm2,%ymm1,%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 f5 74 48 39 c2[ 	]+vcvtbiasps2bf8[ 	]+%zmm2,%zmm1,%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 f5 74 08 39 41 7f[ 	]+vcvtbiasps2bf8[ 	]+0x7f0\(%rcx\),%xmm1,%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 f5 74 28 39 41 7f[ 	]+vcvtbiasps2bf8[ 	]+0xfe0\(%rcx\),%ymm1,%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 f5 74 48 39 41 7f[ 	]+vcvtbiasps2bf8[ 	]+0x1fc0\(%rcx\),%zmm1,%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 d5 74 18 39 01[ 	]+vcvtbiasps2bf8[ 	]+\(%r9\)\{1to4\},%xmm1,%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 d5 74 38 39 01[ 	]+vcvtbiasps2bf8[ 	]+\(%r9\)\{1to8\},%ymm1,%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 d5 74 58 39 01[ 	]+vcvtbiasps2bf8[ 	]+\(%r9\)\{1to16\},%zmm1,%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 f5 74 09 39 c2[ 	]+vcvtbiasps2bf8[ 	]+%xmm2,%xmm1,%xmm0\{%k1\}
[ 	]*[a-f0-9]+:[ 	]*62 f5 74 89 39 c2[ 	]+vcvtbiasps2bf8[ 	]+%xmm2,%xmm1,%xmm0\{%k1\}\{z\}
[ 	]*[a-f0-9]+:[ 	]*62 f5 7e 08 3b c1[ 	]+vcvtps2bf8s[ 	]+%xmm1,%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 f5 7e 28 3b c1[ 	]+vcvtps2bf8s[ 	]+%ymm1,%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 f5 7e 48 3b c1[ 	]+vcvtps2bf8s[ 	]+%zmm1,%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 f5 7e 08 3b 41 7f[ 	]+vcvtps2bf8sx[ 	]+0x7f0\(%rcx\),%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 f5 7e 28 3b 41 7f[ 	]+vcvtps2bf8sy[ 	]+0xfe0\(%rcx\),%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 f5 7e 48 3b 41 7f[ 	]+vcvtps2bf8sz[ 	]+0x1fc0\(%rcx\),%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 d5 7e 18 3b 01[ 	]+vcvtps2bf8s[ 	]+\(%r9\)\{1to4\},%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 d5 7e 38 3b 01[ 	]+vcvtps2bf8s[ 	]+\(%r9\)\{1to8\},%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 d5 7e 58 3b 01[ 	]+vcvtps2bf8s[ 	]+\(%r9\)\{1to16\},%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 f5 7e 09 3b c1[ 	]+vcvtps2bf8s[ 	]+%xmm1,%xmm0\{%k1\}
[ 	]*[a-f0-9]+:[ 	]*62 f5 7e 89 3b c1[ 	]+vcvtps2bf8s[ 	]+%xmm1,%xmm0\{%k1\}\{z\}
[ 	]*[a-f0-9]+:[ 	]*62 f5 74 08 3b c2[ 	]+vcvtbiasps2bf8s[ 	]+%xmm2,%xmm1,%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 f5 74 28 3b c2[ 	]+vcvtbiasps2bf8s[ 	]+%ymm2,%ymm1,%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 f5 74 48 3b c2[ 	]+vcvtbiasps2bf8s[ 	]+%zmm2,%zmm1,%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 f5 74 08 3b 41 7f[ 	]+vcvtbiasps2bf8s[ 	]+0x7f0\(%rcx\),%xmm1,%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 f5 74 28 3b 41 7f[ 	]+vcvtbiasps2bf8s[ 	]+0xfe0\(%rcx\),%ymm1,%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 f5 74 48 3b 41 7f[ 	]+vcvtbiasps2bf8s[ 	]+0x1fc0\(%rcx\),%zmm1,%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 d5 74 18 3b 01[ 	]+vcvtbiasps2bf8s[ 	]+\(%r9\)\{1to4\},%xmm1,%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 d5 74 38 3b 01[ 	]+vcvtbiasps2bf8s[ 	]+\(%r9\)\{1to8\},%ymm1,%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 d5 74 58 3b 01[ 	]+vcvtbiasps2bf8s[ 	]+\(%r9\)\{1to16\},%zmm1,%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 f5 74 09 3b c2[ 	]+vcvtbiasps2bf8s[ 	]+%xmm2,%xmm1,%xmm0\{%k1\}
[ 	]*[a-f0-9]+:[ 	]*62 f5 74 89 3b c2[ 	]+vcvtbiasps2bf8s[ 	]+%xmm2,%xmm1,%xmm0\{%k1\}\{z\}
[ 	]*[a-f0-9]+:[ 	]*62 f5 7e 08 38 c1[ 	]+vcvtps2hf8[ 	]+%xmm1,%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 f5 7e 28 38 c1[ 	]+vcvtps2hf8[ 	]+%ymm1,%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 f5 7e 48 38 c1[ 	]+vcvtps2hf8[ 	]+%zmm1,%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 f5 7e 08 38 41 7f[ 	]+vcvtps2hf8x[ 	]+0x7f0\(%rcx\),%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 f5 7e 28 38 41 7f[ 	]+vcvtps2hf8y[ 	]+0xfe0\(%rcx\),%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 f5 7e 48 38 41 7f[ 	]+vcvtps2hf8z[ 	]+0x1fc0\(%rcx\),%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 d5 7e 18 38 01[ 	]+vcvtps2hf8[ 	]+\(%r9\)\{1to4\},%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 d5 7e 38 38 01[ 	]+vcvtps2hf8[ 	]+\(%r9\)\{1to8\},%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 d5 7e 58 38 01[ 	]+vcvtps2hf8[ 	]+\(%r9\)\{1to16\},%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 f5 7e 09 38 c1[ 	]+vcvtps2hf8[ 	]+%xmm1,%xmm0\{%k1\}
[ 	]*[a-f0-9]+:[ 	]*62 f5 7e 89 38 c1[ 	]+vcvtps2hf8[ 	]+%xmm1,%xmm0\{%k1\}\{z\}
[ 	]*[a-f0-9]+:[ 	]*62 f5 74 08 38 c2[ 	]+vcvtbiasps2hf8[ 	]+%xmm2,%xmm1,%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 f5 74 28 38 c2[ 	]+vcvtbiasps2hf8[ 	]+%ymm2,%ymm1,%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 f5 74 48 38 c2[ 	]+vcvtbiasps2hf8[ 	]+%zmm2,%zmm1,%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 f5 74 08 38 41 7f[ 	]+vcvtbiasps2hf8[ 	]+0x7f0\(%rcx\),%xmm1,%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 f5 74 28 38 41 7f[ 	]+vcvtbiasps2hf8[ 	]+0xfe0\(%rcx\),%ymm1,%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 f5 74 48 38 41 7f[ 	]+vcvtbiasps2hf8[ 	]+0x1fc0\(%rcx\),%zmm1,%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 d5 74 18 38 01[ 	]+vcvtbiasps2hf8[ 	]+\(%r9\)\{1to4\},%xmm1,%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 d5 74 38 38 01[ 	]+vcvtbiasps2hf8[ 	]+\(%r9\)\{1to8\},%ymm1,%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 d5 74 58 38 01[ 	]+vcvtbiasps2hf8[ 	]+\(%r9\)\{1to16\},%zmm1,%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 f5 74 09 38 c2[ 	]+vcvtbiasps2hf8[ 	]+%xmm2,%xmm1,%xmm0\{%k1\}
[ 	]*[a-f0-9]+:[ 	]*62 f5 74 89 38 c2[ 	]+vcvtbiasps2hf8[ 	]+%xmm2,%xmm1,%xmm0\{%k1\}\{z\}
[ 	]*[a-f0-9]+:[ 	]*62 f5 7e 08 3a c1[ 	]+vcvtps2hf8s[ 	]+%xmm1,%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 f5 7e 28 3a c1[ 	]+vcvtps2hf8s[ 	]+%ymm1,%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 f5 7e 48 3a c1[ 	]+vcvtps2hf8s[ 	]+%zmm1,%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 f5 7e 08 3a 41 7f[ 	]+vcvtps2hf8sx[ 	]+0x7f0\(%rcx\),%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 f5 7e 28 3a 41 7f[ 	]+vcvtps2hf8sy[ 	]+0xfe0\(%rcx\),%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 f5 7e 48 3a 41 7f[ 	]+vcvtps2hf8sz[ 	]+0x1fc0\(%rcx\),%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 d5 7e 18 3a 01[ 	]+vcvtps2hf8s[ 	]+\(%r9\)\{1to4\},%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 d5 7e 38 3a 01[ 	]+vcvtps2hf8s[ 	]+\(%r9\)\{1to8\},%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 d5 7e 58 3a 01[ 	]+vcvtps2hf8s[ 	]+\(%r9\)\{1to16\},%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 f5 7e 09 3a c1[ 	]+vcvtps2hf8s[ 	]+%xmm1,%xmm0\{%k1\}
[ 	]*[a-f0-9]+:[ 	]*62 f5 7e 89 3a c1[ 	]+vcvtps2hf8s[ 	]+%xmm1,%xmm0\{%k1\}\{z\}
[ 	]*[a-f0-9]+:[ 	]*62 f5 74 08 3a c2[ 	]+vcvtbiasps2hf8s[ 	]+%xmm2,%xmm1,%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 f5 74 28 3a c2[ 	]+vcvtbiasps2hf8s[ 	]+%ymm2,%ymm1,%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 f5 74 48 3a c2[ 	]+vcvtbiasps2hf8s[ 	]+%zmm2,%zmm1,%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 f5 74 08 3a 41 7f[ 	]+vcvtbiasps2hf8s[ 	]+0x7f0\(%rcx\),%xmm1,%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 f5 74 28 3a 41 7f[ 	]+vcvtbiasps2hf8s[ 	]+0xfe0\(%rcx\),%ymm1,%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 f5 74 48 3a 41 7f[ 	]+vcvtbiasps2hf8s[ 	]+0x1fc0\(%rcx\),%zmm1,%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 d5 74 18 3a 01[ 	]+vcvtbiasps2hf8s[ 	]+\(%r9\)\{1to4\},%xmm1,%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 d5 74 38 3a 01[ 	]+vcvtbiasps2hf8s[ 	]+\(%r9\)\{1to8\},%ymm1,%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 d5 74 58 3a 01[ 	]+vcvtbiasps2hf8s[ 	]+\(%r9\)\{1to16\},%zmm1,%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 f5 74 09 3a c2[ 	]+vcvtbiasps2hf8s[ 	]+%xmm2,%xmm1,%xmm0\{%k1\}
[ 	]*[a-f0-9]+:[ 	]*62 f5 74 89 3a c2[ 	]+vcvtbiasps2hf8s[ 	]+%xmm2,%xmm1,%xmm0\{%k1\}\{z\}
[ 	]*[a-f0-9]+:[ 	]*62 f5 7d 08 38 c1[ 	]+vcvtrops2hf8[ 	]+%xmm1,%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 f5 7d 28 38 c1[ 	]+vcvtrops2hf8[ 	]+%ymm1,%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 f5 7d 48 38 c1[ 	]+vcvtrops2hf8[ 	]+%zmm1,%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 f5 7d 08 38 41 7f[ 	]+vcvtrops2hf8x[ 	]+0x7f0\(%rcx\),%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 f5 7d 28 38 41 7f[ 	]+vcvtrops2hf8y[ 	]+0xfe0\(%rcx\),%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 f5 7d 48 38 41 7f[ 	]+vcvtrops2hf8z[ 	]+0x1fc0\(%rcx\),%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 d5 7d 18 38 01[ 	]+vcvtrops2hf8[ 	]+\(%r9\)\{1to4\},%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 d5 7d 38 38 01[ 	]+vcvtrops2hf8[ 	]+\(%r9\)\{1to8\},%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 d5 7d 58 38 01[ 	]+vcvtrops2hf8[ 	]+\(%r9\)\{1to16\},%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 f5 7d 09 38 c1[ 	]+vcvtrops2hf8[ 	]+%xmm1,%xmm0\{%k1\}
[ 	]*[a-f0-9]+:[ 	]*62 f5 7d 89 38 c1[ 	]+vcvtrops2hf8[ 	]+%xmm1,%xmm0\{%k1\}\{z\}
[ 	]*[a-f0-9]+:[ 	]*62 f5 7d 08 3a c1[ 	]+vcvtrops2hf8s[ 	]+%xmm1,%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 f5 7d 28 3a c1[ 	]+vcvtrops2hf8s[ 	]+%ymm1,%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 f5 7d 48 3a c1[ 	]+vcvtrops2hf8s[ 	]+%zmm1,%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 f5 7d 08 3a 41 7f[ 	]+vcvtrops2hf8sx[ 	]+0x7f0\(%rcx\),%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 f5 7d 28 3a 41 7f[ 	]+vcvtrops2hf8sy[ 	]+0xfe0\(%rcx\),%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 f5 7d 48 3a 41 7f[ 	]+vcvtrops2hf8sz[ 	]+0x1fc0\(%rcx\),%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 d5 7d 18 3a 01[ 	]+vcvtrops2hf8s[ 	]+\(%r9\)\{1to4\},%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 d5 7d 38 3a 01[ 	]+vcvtrops2hf8s[ 	]+\(%r9\)\{1to8\},%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 d5 7d 58 3a 01[ 	]+vcvtrops2hf8s[ 	]+\(%r9\)\{1to16\},%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 f5 7d 09 3a c1[ 	]+vcvtrops2hf8s[ 	]+%xmm1,%xmm0\{%k1\}
[ 	]*[a-f0-9]+:[ 	]*62 f5 7d 89 3a c1[ 	]+vcvtrops2hf8s[ 	]+%xmm1,%xmm0\{%k1\}\{z\}
[ 	]*[a-f0-9]+:[ 	]*62 f5 fc 08 36 c1[ 	]+vcvtbf82ps[ 	]+%xmm1,%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 f5 fc 28 36 c1[ 	]+vcvtbf82ps[ 	]+%xmm1,%ymm0
[ 	]*[a-f0-9]+:[ 	]*62 f5 fc 48 36 c1[ 	]+vcvtbf82ps[ 	]+%xmm1,%zmm0
[ 	]*[a-f0-9]+:[ 	]*62 f5 fc 08 36 41 7f[ 	]+vcvtbf82ps[ 	]+0x1fc\(%rcx\),%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 f5 fc 28 36 41 7f[ 	]+vcvtbf82ps[ 	]+0x3f8\(%rcx\),%ymm0
[ 	]*[a-f0-9]+:[ 	]*62 f5 fc 48 36 41 7f[ 	]+vcvtbf82ps[ 	]+0x7f0\(%rcx\),%zmm0
[ 	]*[a-f0-9]+:[ 	]*62 f5 fc 09 36 c1[ 	]+vcvtbf82ps[ 	]+%xmm1,%xmm0\{%k1\}
[ 	]*[a-f0-9]+:[ 	]*62 f5 fc 89 36 c1[ 	]+vcvtbf82ps[ 	]+%xmm1,%xmm0\{%k1\}\{z\}
[ 	]*[a-f0-9]+:[ 	]*62 f5 fe 08 3d c8[ 	]+vcvtbf82bf4s[ 	]+%xmm1,%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 f5 fe 28 3d c8[ 	]+vcvtbf82bf4s[ 	]+%ymm1,%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 f5 fe 48 3d c8[ 	]+vcvtbf82bf4s[ 	]+%zmm1,%ymm0
[ 	]*[a-f0-9]+:[ 	]*62 f5 fe 08 3d 49 7f[ 	]+vcvtbf82bf4s[ 	]+%xmm1,0x3f8\(%rcx\)
[ 	]*[a-f0-9]+:[ 	]*62 f5 fe 28 3d 49 7f[ 	]+vcvtbf82bf4s[ 	]+%ymm1,0x7f0\(%rcx\)
[ 	]*[a-f0-9]+:[ 	]*62 f5 fe 48 3d 49 7f[ 	]+vcvtbf82bf4s[ 	]+%zmm1,0xfe0\(%rcx\)
[ 	]*[a-f0-9]+:[ 	]*62 f5 7c 08 36 c1[ 	]+vcvthf82ps[ 	]+%xmm1,%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 f5 7c 28 36 c1[ 	]+vcvthf82ps[ 	]+%xmm1,%ymm0
[ 	]*[a-f0-9]+:[ 	]*62 f5 7c 48 36 c1[ 	]+vcvthf82ps[ 	]+%xmm1,%zmm0
[ 	]*[a-f0-9]+:[ 	]*62 f5 7c 08 36 41 7f[ 	]+vcvthf82ps[ 	]+0x1fc\(%rcx\),%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 f5 7c 28 36 41 7f[ 	]+vcvthf82ps[ 	]+0x3f8\(%rcx\),%ymm0
[ 	]*[a-f0-9]+:[ 	]*62 f5 7c 48 36 41 7f[ 	]+vcvthf82ps[ 	]+0x7f0\(%rcx\),%zmm0
[ 	]*[a-f0-9]+:[ 	]*62 f5 7c 09 36 c1[ 	]+vcvthf82ps[ 	]+%xmm1,%xmm0\{%k1\}
[ 	]*[a-f0-9]+:[ 	]*62 f5 7c 89 36 c1[ 	]+vcvthf82ps[ 	]+%xmm1,%xmm0\{%k1\}\{z\}
[ 	]*[a-f0-9]+:[ 	]*62 f5 7e 08 3d c8[ 	]+vcvthf82bf4s[ 	]+%xmm1,%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 f5 7e 28 3d c8[ 	]+vcvthf82bf4s[ 	]+%ymm1,%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 f5 7e 48 3d c8[ 	]+vcvthf82bf4s[ 	]+%zmm1,%ymm0
[ 	]*[a-f0-9]+:[ 	]*62 f5 7e 08 3d 49 7f[ 	]+vcvthf82bf4s[ 	]+%xmm1,0x3f8\(%rcx\)
[ 	]*[a-f0-9]+:[ 	]*62 f5 7e 28 3d 49 7f[ 	]+vcvthf82bf4s[ 	]+%ymm1,0x7f0\(%rcx\)
[ 	]*[a-f0-9]+:[ 	]*62 f5 7e 48 3d 49 7f[ 	]+vcvthf82bf4s[ 	]+%zmm1,0xfe0\(%rcx\)
[ 	]*[a-f0-9]+:[ 	]*62 f5 fe 08 3e c1[ 	]+vcvtbf82bf6s[ 	]+%xmm1,%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 f5 fe 28 3e c1[ 	]+vcvtbf82bf6s[ 	]+%ymm1,%ymm0
[ 	]*[a-f0-9]+:[ 	]*62 f5 fe 48 3e c1[ 	]+vcvtbf82bf6s[ 	]+%zmm1,%zmm0
[ 	]*[a-f0-9]+:[ 	]*62 f5 7e 08 3c c1[ 	]+vcvthf82hf6s[ 	]+%xmm1,%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 f5 7e 28 3c c1[ 	]+vcvthf82hf6s[ 	]+%ymm1,%ymm0
[ 	]*[a-f0-9]+:[ 	]*62 f5 7e 48 3c c1[ 	]+vcvthf82hf6s[ 	]+%zmm1,%zmm0
[ 	]*[a-f0-9]+:[ 	]*62 f5 7c 08 37 c1[ 	]+vcvtbf42hf8[ 	]+%xmm1,%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 f5 7c 28 37 c1[ 	]+vcvtbf42hf8[ 	]+%xmm1,%ymm0
[ 	]*[a-f0-9]+:[ 	]*62 f5 7c 48 37 c1[ 	]+vcvtbf42hf8[ 	]+%ymm1,%zmm0
[ 	]*[a-f0-9]+:[ 	]*62 f5 7c 08 37 41 7f[ 	]+vcvtbf42hf8[ 	]+0x3f8\(%rcx\),%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 f5 7c 28 37 41 7f[ 	]+vcvtbf42hf8[ 	]+0x7f0\(%rcx\),%ymm0
[ 	]*[a-f0-9]+:[ 	]*62 f5 7c 48 37 41 7f[ 	]+vcvtbf42hf8[ 	]+0xfe0\(%rcx\),%zmm0
[ 	]*[a-f0-9]+:[ 	]*62 f5 7c 09 37 c1[ 	]+vcvtbf42hf8[ 	]+%xmm1,%xmm0\{%k1\}
[ 	]*[a-f0-9]+:[ 	]*62 f5 7c 89 37 c1[ 	]+vcvtbf42hf8[ 	]+%xmm1,%xmm0\{%k1\}\{z\}
[ 	]*[a-f0-9]+:[ 	]*62 f5 fd 08 37 c1[ 	]+vcvtbf62hf8[ 	]+%xmm1,%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 f5 fd 28 37 c1[ 	]+vcvtbf62hf8[ 	]+%ymm1,%ymm0
[ 	]*[a-f0-9]+:[ 	]*62 f5 fd 48 37 c1[ 	]+vcvtbf62hf8[ 	]+%zmm1,%zmm0
[ 	]*[a-f0-9]+:[ 	]*62 f5 fd 09 37 c1[ 	]+vcvtbf62hf8[ 	]+%xmm1,%xmm0\{%k1\}
[ 	]*[a-f0-9]+:[ 	]*62 f5 fd 89 37 c1[ 	]+vcvtbf62hf8[ 	]+%xmm1,%xmm0\{%k1\}\{z\}
[ 	]*[a-f0-9]+:[ 	]*62 f5 7d 08 37 c1[ 	]+vcvthf62hf8[ 	]+%xmm1,%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 f5 7d 28 37 c1[ 	]+vcvthf62hf8[ 	]+%ymm1,%ymm0
[ 	]*[a-f0-9]+:[ 	]*62 f5 7d 48 37 c1[ 	]+vcvthf62hf8[ 	]+%zmm1,%zmm0
[ 	]*[a-f0-9]+:[ 	]*62 f5 7d 09 37 c1[ 	]+vcvthf62hf8[ 	]+%xmm1,%xmm0\{%k1\}
[ 	]*[a-f0-9]+:[ 	]*62 f5 7d 89 37 c1[ 	]+vcvthf62hf8[ 	]+%xmm1,%xmm0\{%k1\}\{z\}
[ 	]*[a-f0-9]+:[ 	]*62 f2 7e 08 41 c8[ 	]+vpmovssdb[ 	]+%xmm1,%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 f2 7e 28 41 c8[ 	]+vpmovssdb[ 	]+%ymm1,%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 f2 7e 48 41 c8[ 	]+vpmovssdb[ 	]+%zmm1,%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 f2 7e 08 41 49 7f[ 	]+vpmovssdb[ 	]+%xmm1,0x1fc\(%rcx\)
[ 	]*[a-f0-9]+:[ 	]*62 f2 7e 28 41 49 7f[ 	]+vpmovssdb[ 	]+%ymm1,0x3f8\(%rcx\)
[ 	]*[a-f0-9]+:[ 	]*62 f2 7e 48 41 49 7f[ 	]+vpmovssdb[ 	]+%zmm1,0x7f0\(%rcx\)
[ 	]*[a-f0-9]+:[ 	]*62 f2 7e 09 41 c8[ 	]+vpmovssdb[ 	]+%xmm1,%xmm0\{%k1\}
[ 	]*[a-f0-9]+:[ 	]*62 f2 7e 89 41 c8[ 	]+vpmovssdb[ 	]+%xmm1,%xmm0\{%k1\}\{z\}
[ 	]*[a-f0-9]+:[ 	]*62 f3 7c 08 3d c1 10[ 	]+vunpackb[ 	]+\$0x10,%xmm1,%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 f3 7c 28 3d c1 10[ 	]+vunpackb[ 	]+\$0x10,%ymm1,%ymm0
[ 	]*[a-f0-9]+:[ 	]*62 f3 7c 48 3d c1 10[ 	]+vunpackb[ 	]+\$0x10,%zmm1,%zmm0
[ 	]*[a-f0-9]+:[ 	]*62 f3 7c 08 3d 41 7f 10[ 	]+vunpackb[ 	]+\$0x10,0x7f0\(%rcx\),%xmm0
[ 	]*[a-f0-9]+:[ 	]*62 f3 7c 28 3d 41 7f 10[ 	]+vunpackb[ 	]+\$0x10,0xfe0\(%rcx\),%ymm0
[ 	]*[a-f0-9]+:[ 	]*62 f3 7c 48 3d 41 7f 10[ 	]+vunpackb[ 	]+\$0x10,0x1fc0\(%rcx\),%zmm0
[ 	]*[a-f0-9]+:[ 	]*62 f3 7c 09 3d c1 10[ 	]+vunpackb[ 	]+\$0x10,%xmm1,%xmm0\{%k1\}
[ 	]*[a-f0-9]+:[ 	]*62 f3 7c 89 3d c1 10[ 	]+vunpackb[ 	]+\$0x10,%xmm1,%xmm0\{%k1\}\{z\}
#pass
