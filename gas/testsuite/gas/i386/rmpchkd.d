#as: -march=generic64+rmpdirty
#objdump: -dw
#name: 64-bit RMPCHKD insn
#source: rmpchkd.s

.*: +file format .*


Disassembly of section \.text:

0+ <att>:
[ 	]*[a-f0-9]+:[ 	]+f3 0f 01 fc[ 	]+rmpchkd[ 	]*
[ 	]*[a-f0-9]+:[ 	]+f3 0f 01 fc[ 	]+rmpchkd[ 	]*
[ 	]*[a-f0-9]+:[ 	]+67 f3 0f 01 fc[ 	]+addr32 rmpchkd[ 	]*

[0-9a-f]+ <intel>:
[ 	]*[a-f0-9]+:[ 	]+f3 0f 01 fc[ 	]+rmpchkd[ 	]*
[ 	]*[a-f0-9]+:[ 	]+f3 0f 01 fc[ 	]+rmpchkd[ 	]*
[ 	]*[a-f0-9]+:[ 	]+67 f3 0f 01 fc[ 	]+addr32 rmpchkd[ 	]*
#pass
