#source: bad-bcast.s
#objdump: -dw -Mintel
#name: Disassemble bad broadcast (Intel mode)

.*: +file format .*

Disassembly of section .text:

0+ <.text>:
[ 	]*[a-f0-9]+:[ 	]*62 c3 8c 1d 66\s*\(bad\)
[ 	]*[a-f0-9]+:[ 	]*90\s*nop
[ 	]*[a-f0-9]+:[ 	]*66 90\s*xchg   ax,ax
[ 	]*[a-f0-9]+:[ 	]*66 90\s*xchg   ax,ax
[ 	]*[a-f0-9]+:[ 	]*62 c1 ff 38 2a 20\s*vcvtsi2sd xmm4,xmm0,\[eax\]{bad}
[ 	]*[a-f0-9]+:[ 	]*62 f1 fe 58 7f 09\s*vmovdqu64 \[ecx\]{bad},zmm1
[ 	]*[a-f0-9]+:[ 	]*62 f1 7f 58 6f 09\s*vmovdqu8 zmm1,\[ecx\]{bad}
[ 	]*[a-f0-9]+:[ 	]*62 f1 ff 58 6f 09\s*vmovdqu16 zmm1,\[ecx\]{bad}
[ 	]*[a-f0-9]+:[ 	]*62 f1 fe 58 6f 09\s*vmovdqu64 zmm1,\[ecx\]{bad}
#pass
