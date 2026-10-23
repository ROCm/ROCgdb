#as: -EL -mdialect=normal
#objdump: -dr
#source: hmul.s
#name: BPF UHMUL and SHMUL insns, normal syntax

.*: +file format .*bpf.*

Disassembly of section .text:

0+ <.text>:
   0:	2f 12 01 00 00 00 00 00 	uhmul %r2,%r1
   8:	27 02 01 00 21 00 00 00 	uhmul %r2,33
  10:	27 02 01 00 f9 ff ff ff 	uhmul %r2,-7
  18:	2f 25 02 00 00 00 00 00 	shmul %r5,%r2
  20:	27 05 02 00 09 00 00 00 	shmul %r5,9
  28:	27 05 02 00 fd ff ff ff 	shmul %r5,-3
