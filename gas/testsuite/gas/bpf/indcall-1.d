#as: -EL -misa-spec=xbpf
#objdump: -dr -M xbpf,dec
#source: indcall-1.s
#name: BPF indirect call 1, normal syntax

.*: +file format .*bpf.*

Disassembly of section \.text:

0000000000000000 <main>:
   0:	b7 00 00 00 01 00 00 00 	mov %r0,1
   8:	b7 01 00 00 01 00 00 00 	mov %r1,1
  10:	b7 02 00 00 02 00 00 00 	mov %r2,2
  18:	18 06 00 00 38 00 00 00 	lddw %r6,56
  20:	00 00 00 00 00 00 00 00[    ]*
			18: R_BPF_64_64	.text
  28:	8d 06 00 00 00 00 00 00 	call %r6
  30:	95 00 00 00 00 00 00 00 	exit

0000000000000038 <bar>:
  38:	b7 00 00 00 00 00 00 00 	mov %r0,0
  40:	95 00 00 00 00 00 00 00 	exit
#pass
