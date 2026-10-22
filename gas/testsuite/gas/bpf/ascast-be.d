#as: -EB -mdialect=normal
#objdump: -dr -M dec
#source: ascast.s
#name: BPF address space cast instructions, normal syntax, big endian

.*: +file format .*bpf.*

Disassembly of section .text:

0+ <.text>:
   0:	bf 11 00 01 00 00 00 01 	addrc %r1,%r1,0,1
   8:	bf 01 00 01 00 01 00 00 	addrc %r0,%r1,1,0
  10:	bf 92 00 01 00 00 00 01 	addrc %r9,%r2,0,1
  18:	bf 23 00 00 00 00 00 00 	mov %r2,%r3
  20:	bf 34 00 01 00 00 00 01 	addrc %r3,%r4,0,1
  28:	bf 56 00 01 00 01 00 00 	addrc %r5,%r6,1,0
