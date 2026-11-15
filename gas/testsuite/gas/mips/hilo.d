#as: -32
#objdump: -drz --prefix-addresses --show-raw-insn
#name: MIPS HI/LO moves

.*: +file format .*mips.*

Disassembly of section \.text:
[0-9a-f]+ <[^>]*> 00001012 	mflo	v0
[0-9a-f]+ <[^>]*> 00600013 	mtlo	v1
[0-9a-f]+ <[^>]*> 00002010 	mfhi	a0
[0-9a-f]+ <[^>]*> 00a00011 	mthi	a1
[0-9a-f]+ <[^>]*> 00003012 	mflo	a2
[0-9a-f]+ <[^>]*> 00e00011 	mthi	a3
[0-9a-f]+ <[^>]*> 00008010 	mfhi	s0
[0-9a-f]+ <[^>]*> 02200013 	mtlo	s1
[0-9a-f]+ <[^>]*> 00009012 	mflo	s2
#pass
