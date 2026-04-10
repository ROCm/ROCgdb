#as: -mrelax-relocations=no
#objdump: -dwr

.*: +file format .*


Disassembly of section .text:

0+ <foo>:
 +[a-f0-9]+:	48 8b 05 00 00 00 00 	mov    0x0\(%rip\),%rax        # 7 <foo\+0x7>	3: R_X86_64_GOTPCREL	foo-0x4
 +[a-f0-9]+:	ff 35 00 00 00 00    	push   0x0\(%rip\)        # .*	9: R_X86_64_GOTPCREL	foo\+0x4
 +[a-f0-9]+:	ff 35 00 00 00 00    	push   0x0\(%rip\)        # .*	f: R_X86_64_GOTPCREL	foo-0xc
#pass
