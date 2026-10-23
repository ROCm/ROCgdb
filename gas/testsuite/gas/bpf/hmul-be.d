#as: -EB -mdialect=normal
#objdump: -dr
#source: hmul.s
#name: BPF UHMUL and SHMUL insns, normal syntax, big endian

.*: +file format .*bpf.*

Disassembly of section .text:

0+ <.text>:
   0:	2f 21 00 01 00 00 00 00 	uhmul %r2,%r1
   8:	27 20 00 01 00 00 00 21 	uhmul %r2,33
  10:	27 20 00 01 ff ff ff f9 	uhmul %r2,-7
  18:	2f 52 00 02 00 00 00 00 	shmul %r5,%r2
  20:	27 50 00 02 00 00 00 09 	shmul %r5,9
  28:	27 50 00 02 ff ff ff fd 	shmul %r5,-3
