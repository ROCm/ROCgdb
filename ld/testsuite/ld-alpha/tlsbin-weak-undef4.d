#ld: -z norelro -z nomemory-seal
#objdump: -dw

.*: +file format .*


Disassembly of section .text:

[a-f0-9]+ <_start>:
 +[a-f0-9]+:	00 00 00 24 	ldah	v0,0\(v0\)
 +[a-f0-9]+:	00 00 20 a0 	ldl	t0,0\(v0\)
 +[a-f0-9]+:	01 80 fa 6b 	ret
 +[a-f0-9]+:	00 00 fe 2f 	unop	
#pass
