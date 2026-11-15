#as: -32
#objdump: -drz --prefix-addresses --show-raw-insn
#name: MIPS HI/LO moves
#source: hilo.s

.*: +file format .*mips.*

Disassembly of section \.text:
[0-9a-f]+ <[^>]*> 4642      	mflo	v0
[0-9a-f]+ <[^>]*> 0003 3d7c 	mtlo	v1
[0-9a-f]+ <[^>]*> 4604      	mfhi	a0
[0-9a-f]+ <[^>]*> 0005 2d7c 	mthi	a1
[0-9a-f]+ <[^>]*> 4646      	mflo	a2
[0-9a-f]+ <[^>]*> 0007 2d7c 	mthi	a3
[0-9a-f]+ <[^>]*> 4610      	mfhi	s0
[0-9a-f]+ <[^>]*> 0011 3d7c 	mtlo	s1
[0-9a-f]+ <[^>]*> 4652      	mflo	s2
#pass
