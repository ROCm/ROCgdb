#objdump: -dw
#name: i386 AVX10_VPMLDSABFLYDQ insns

.*: +file format .*

Disassembly of section \.text:

0+ <_start>:
[ 	]*[a-f0-9]+:[ 	]*62 f3 d5 08 dc f4 7b[ 	]*vpmldsabflydq \$0x7b,%xmm4,%xmm5,%xmm6
[ 	]*[a-f0-9]+:[ 	]*62 f3 d5 28 dc f4 7b[ 	]*vpmldsabflydq \$0x7b,%ymm4,%ymm5,%ymm6
[ 	]*[a-f0-9]+:[ 	]*62 f3 d5 48 dc f4 7b[ 	]*vpmldsabflydq \$0x7b,%zmm4,%zmm5,%zmm6
[ 	]*[a-f0-9]+:[ 	]*62 f3 d5 08 dc b4 f4 00 00 00 10 7b[ 	]*vpmldsabflydq \$0x7b,0x10000000\(%esp,%esi,8\),%xmm5,%xmm6
[ 	]*[a-f0-9]+:[ 	]*62 f3 d5 08 dc 31 7b[ 	]*vpmldsabflydq \$0x7b,\(%ecx\),%xmm5,%xmm6
[ 	]*[a-f0-9]+:[ 	]*62 f3 d5 08 dc 71 7f 7b[ 	]*vpmldsabflydq \$0x7b,0x7f0\(%ecx\),%xmm5,%xmm6
[ 	]*[a-f0-9]+:[ 	]*62 f3 d5 08 dc 72 80 7b[ 	]*vpmldsabflydq \$0x7b,-0x800\(%edx\),%xmm5,%xmm6
[ 	]*[a-f0-9]+:[ 	]*62 f3 d5 28 dc b4 f4 00 00 00 10 7b[ 	]*vpmldsabflydq \$0x7b,0x10000000\(%esp,%esi,8\),%ymm5,%ymm6
[ 	]*[a-f0-9]+:[ 	]*62 f3 d5 28 dc 31 7b[ 	]*vpmldsabflydq \$0x7b,\(%ecx\),%ymm5,%ymm6
[ 	]*[a-f0-9]+:[ 	]*62 f3 d5 28 dc 71 7f 7b[ 	]*vpmldsabflydq \$0x7b,0xfe0\(%ecx\),%ymm5,%ymm6
[ 	]*[a-f0-9]+:[ 	]*62 f3 d5 28 dc 72 80 7b[ 	]*vpmldsabflydq \$0x7b,-0x1000\(%edx\),%ymm5,%ymm6
[ 	]*[a-f0-9]+:[ 	]*62 f3 d5 48 dc b4 f4 00 00 00 10 7b[ 	]*vpmldsabflydq \$0x7b,0x10000000\(%esp,%esi,8\),%zmm5,%zmm6
[ 	]*[a-f0-9]+:[ 	]*62 f3 d5 48 dc 31 7b[ 	]*vpmldsabflydq \$0x7b,\(%ecx\),%zmm5,%zmm6
[ 	]*[a-f0-9]+:[ 	]*62 f3 d5 48 dc 71 7f 7b[ 	]*vpmldsabflydq \$0x7b,0x1fc0\(%ecx\),%zmm5,%zmm6
[ 	]*[a-f0-9]+:[ 	]*62 f3 d5 48 dc 72 80 7b[ 	]*vpmldsabflydq \$0x7b,-0x2000\(%edx\),%zmm5,%zmm6
#pass
