#ld: -shared -z norelro -z nomemory-seal
#objdump: -dw

.*: +file format .*


Disassembly of section .text:

[a-f0-9]+ <_start>:
 +[a-f0-9]+:	00 80 3d a4 	ldq	t0,-32768\(gp\)
 +[a-f0-9]+:	01 04 01 40 	addq	v0,t0,t0
 +[a-f0-9]+:	01 80 fa 6b 	ret
 +[a-f0-9]+:	00 00 fe 2f 	unop	
#pass
