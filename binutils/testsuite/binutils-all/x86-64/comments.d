#name: disassembly comments (AT&T)
#ld:
#objdump: -dwMannotate,att

.*: +file format .*


Disassembly of section .text:

[0-9a-f]+ <_start>:
[ 	]*[0-9a-f]+:	48 c7 05 [0-9a-f ]+	movq   \$0x[0-9a-f]+,0x[0-9a-f]+\(%rip\) +# \[_start\], [0-9a-f]+ <fptr>
[ 	]*[0-9a-f]+:	31 c0                	xor    %eax,%eax
[ 	]*[0-9a-f]+:	c3                   	ret
#pass
