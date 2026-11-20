#objdump: -dw
#name: x86_64 AVX10_VPMLDSABFLYDQ insns

.*: +file format .*

Disassembly of section \.text:

0+ <_start>:
[ 	]*[a-f0-9]+:[ 	]*62 03 95 00 dc f4 7b[ 	]*vpmldsabflydq \$0x7b,%xmm28,%xmm29,%xmm30
[ 	]*[a-f0-9]+:[ 	]*62 03 95 20 dc f4 7b[ 	]*vpmldsabflydq \$0x7b,%ymm28,%ymm29,%ymm30
[ 	]*[a-f0-9]+:[ 	]*62 03 95 40 dc f4 7b[ 	]*vpmldsabflydq \$0x7b,%zmm28,%zmm29,%zmm30
[ 	]*[a-f0-9]+:[ 	]*62 23 95 00 dc b4 f5 00 00 00 10 7b[ 	]*vpmldsabflydq \$0x7b,0x10000000\(%rbp,%r14,8\),%xmm29,%xmm30
[ 	]*[a-f0-9]+:[ 	]*62 43 95 00 dc 31 7b[ 	]*vpmldsabflydq \$0x7b,\(%r9\),%xmm29,%xmm30
[ 	]*[a-f0-9]+:[ 	]*62 63 95 00 dc 71 7f 7b[ 	]*vpmldsabflydq \$0x7b,0x7f0\(%rcx\),%xmm29,%xmm30
[ 	]*[a-f0-9]+:[ 	]*62 63 95 00 dc 72 80 7b[ 	]*vpmldsabflydq \$0x7b,-0x800\(%rdx\),%xmm29,%xmm30
[ 	]*[a-f0-9]+:[ 	]*62 23 95 20 dc b4 f5 00 00 00 10 7b[ 	]*vpmldsabflydq \$0x7b,0x10000000\(%rbp,%r14,8\),%ymm29,%ymm30
[ 	]*[a-f0-9]+:[ 	]*62 43 95 20 dc 31 7b[ 	]*vpmldsabflydq \$0x7b,\(%r9\),%ymm29,%ymm30
[ 	]*[a-f0-9]+:[ 	]*62 63 95 20 dc 71 7f 7b[ 	]*vpmldsabflydq \$0x7b,0xfe0\(%rcx\),%ymm29,%ymm30
[ 	]*[a-f0-9]+:[ 	]*62 63 95 20 dc 72 80 7b[ 	]*vpmldsabflydq \$0x7b,-0x1000\(%rdx\),%ymm29,%ymm30
[ 	]*[a-f0-9]+:[ 	]*62 23 95 40 dc b4 f5 00 00 00 10 7b[ 	]*vpmldsabflydq \$0x7b,0x10000000\(%rbp,%r14,8\),%zmm29,%zmm30
[ 	]*[a-f0-9]+:[ 	]*62 43 95 40 dc 31 7b[ 	]*vpmldsabflydq \$0x7b,\(%r9\),%zmm29,%zmm30
[ 	]*[a-f0-9]+:[ 	]*62 63 95 40 dc 71 7f 7b[ 	]*vpmldsabflydq \$0x7b,0x1fc0\(%rcx\),%zmm29,%zmm30
[ 	]*[a-f0-9]+:[ 	]*62 63 95 40 dc 72 80 7b[ 	]*vpmldsabflydq \$0x7b,-0x2000\(%rdx\),%zmm29,%zmm30
#pass
