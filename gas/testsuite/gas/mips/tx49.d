#objdump: -dr --prefix-addresses --show-raw-insn -mmips:tx49 -M gpr-names=32
#name: MIPS TX49 instructions
#as: -march=tx49

# Check MIPS TX49 instruction assembly

.*: +file format .*mips.*

Disassembly of section .text:
0+0000 <[^>]*> 70220000 	madd	at,v0
0+0004 <[^>]*> 70851800 	madd	v1,a0,a1
0+0008 <[^>]*> 70c70001 	maddu	a2,a3
0+000c <[^>]*> 712a4001 	maddu	t0,t1,t2
0+0010 <[^>]*> 018d5818 	mult	t3,t4,t5
0+0014 <[^>]*> 01f07019 	multu	t6,t7,s0
0+0018 <[^>]*> 0253881c 	dmult	s1,s2,s3
0+001c <[^>]*> 02b6a01d 	dmultu	s4,s5,s6
0+0020 <[^>]*> cee40000 	pref	0x4,0\(s7\)
0+0024 <[^>]*> cf047fff 	pref	0x4,32767\(t8\)
0+0028 <[^>]*> 40d9d000 	ctc0	t9,\$26
0+002c <[^>]*> 405be000 	cfc0	k1,\$28
0+0030 <[^>]*> 42000020 	wait
0+0034 <[^>]*> 0000000e 	sdbbp
0+0038 <[^>]*> 0056000e 	sdbbp	0x56
0+003c <[^>]*> 4200001f 	deret
	\.\.\.
