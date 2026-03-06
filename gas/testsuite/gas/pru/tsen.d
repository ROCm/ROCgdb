#objdump: -dr --prefix-addresses --show-raw-insn
#name: PRU tsen
#as: -mcore-revision=V4

# Test the TSEN instruction

.*: +file format elf32-pru

Disassembly of section .text:
0+0000 <[^>]*> 32000000 	tsen	0
0+0004 <[^>]*> 32800000 	tsen	1
