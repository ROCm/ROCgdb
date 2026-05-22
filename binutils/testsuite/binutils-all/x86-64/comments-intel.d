#name: disassembly comments (Intel)
#source: comments.s
#ld:
#objdump: -dwMannotate,intel

.*: +file format .*


Disassembly of section .text:

[0-9a-f]+ <_start>:
[ 	]*[0-9a-f]+:	48 c7 05 [0-9a-f ]+	mov    QWORD PTR \[rip\+0x[0-9a-f]+\],0x[0-9a-f]+ +# [0-9a-f]+ <fptr>, \[_start\]
[ 	]*[0-9a-f]+:	31 c0                	xor    eax,eax
[ 	]*[0-9a-f]+:	c3                   	ret
