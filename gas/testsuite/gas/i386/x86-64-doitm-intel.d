#objdump: -dw -Mintel
#name: x86_64 DOITM insns (Intel disassembly)
#source: x86-64-doitm.s

.*: +file format .*

Disassembly of section \.text:

#...
[a-f0-9]+ <_intel>:
[ 	]*[a-f0-9]+:[ 	]*f3 49 0f 1e f4[ 	]*rddoitm r12
[ 	]*[a-f0-9]+:[ 	]*f3 48 0f 1e f3[ 	]*rddoitm rbx
[ 	]*[a-f0-9]+:[ 	]*f3 d5 98 1e f6[ 	]*rddoitm r22
[ 	]*[a-f0-9]+:[ 	]*f3 0f 1e fc[ 	]*setdoitm
[ 	]*[a-f0-9]+:[ 	]*f2 49 0f 1e f6[ 	]*wrdoitm r14
[ 	]*[a-f0-9]+:[ 	]*f2 48 0f 1e f1[ 	]*wrdoitm rcx
[ 	]*[a-f0-9]+:[ 	]*f2 d5 99 1e f0[ 	]*wrdoitm r24
#pass
