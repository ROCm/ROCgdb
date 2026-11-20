#objdump: -dw -Mintel
#name: i386 DOITM insns (Intel disassembly)
#source: doitm.s

.*: +file format .*

Disassembly of section \.text:

#...
[a-f0-9]+ <_intel>:
[ 	]*[a-f0-9]+:[ 	]*f3 0f 1e f0[ 	]*rddoitm eax
[ 	]*[a-f0-9]+:[ 	]*f3 0f 1e fc[ 	]*setdoitm
[ 	]*[a-f0-9]+:[ 	]*f2 0f 1e f3[ 	]*wrdoitm ebx
#pass
