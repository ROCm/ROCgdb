#objdump: -dw
#name: i386 DOITM insns

.*: +file format .*

Disassembly of section \.text:

0+ <_start>:
[ 	]*[a-f0-9]+:[ 	]*f3 0f 1e f0[ 	]*rddoitm %eax
[ 	]*[a-f0-9]+:[ 	]*f3 0f 1e fc[ 	]*setdoitm
[ 	]*[a-f0-9]+:[ 	]*f2 0f 1e f3[ 	]*wrdoitm %ebx
#pass
