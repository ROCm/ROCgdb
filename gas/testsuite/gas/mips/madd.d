#as: -32
#objdump: -drz --prefix-addresses --show-raw-insn
#name: MIPS integer multiply-add hazard

.*: +file format .*mips.*

Disassembly of section \.text:
[0-9a-f]+ <[^>]*> 70850000 	madd	a0,a1
[0-9a-f]+ <[^>]*> 00000000 	nop
[0-9a-f]+ <[^>]*> 00000000 	nop
[0-9a-f]+ <[^>]*> 00c7001a 	div	zero,a2,a3
[0-9a-f]+ <[^>]*> 70851000 	madd	v0,a0,a1
[0-9a-f]+ <[^>]*> 00000000 	nop
[0-9a-f]+ <[^>]*> 00000000 	nop
[0-9a-f]+ <[^>]*> 00c7001a 	div	zero,a2,a3
[0-9a-f]+ <[^>]*> 70850001 	maddu	a0,a1
[0-9a-f]+ <[^>]*> 00000000 	nop
[0-9a-f]+ <[^>]*> 00000000 	nop
[0-9a-f]+ <[^>]*> 00c7001b 	divu	zero,a2,a3
[0-9a-f]+ <[^>]*> 70851001 	maddu	v0,a0,a1
[0-9a-f]+ <[^>]*> 00000000 	nop
[0-9a-f]+ <[^>]*> 00000000 	nop
[0-9a-f]+ <[^>]*> 00c7001b 	divu	zero,a2,a3
#pass
