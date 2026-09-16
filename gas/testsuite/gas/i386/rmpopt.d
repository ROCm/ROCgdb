#as: -march=generic64+rmpopt
#objdump: -dw
#name: 64-bit RMPOPT insn
#source: rmpopt.s

.*: +file format .*


Disassembly of section \.text:

0+ <att>:
[ 	]*[a-f0-9]+:[ 	]+f2 0f 01 fc[ 	]+rmpopt[ 	]*
[ 	]*[a-f0-9]+:[ 	]+f2 0f 01 fc[ 	]+rmpopt[ 	]*
[ 	]*[a-f0-9]+:[ 	]+67 f2 0f 01 fc[ 	]+addr32 rmpopt[ 	]*

[0-9a-f]+ <intel>:
[ 	]*[a-f0-9]+:[ 	]+f2 0f 01 fc[ 	]+rmpopt[ 	]*
[ 	]*[a-f0-9]+:[ 	]+f2 0f 01 fc[ 	]+rmpopt[ 	]*
[ 	]*[a-f0-9]+:[ 	]+67 f2 0f 01 fc[ 	]+addr32 rmpopt[ 	]*
#pass
