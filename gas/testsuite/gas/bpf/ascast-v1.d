#as: -EL -misa-spec=v1 -mdialect=normal
#objdump: -dr -M dec,v1
#source: ascast.s
#name: BPF address space cast instructions, ISA v1

.*: +file format .*bpf.*

Disassembly of section .text:

0+ <.text>:
   0:	bf 11 01 00 01 00 00 00 	addrc %r1,%r1,0,1
   8:	bf 10 01 00 00 00 01 00 	addrc %r0,%r1,1,0
  10:	bf 29 01 00 01 00 00 00 	addrc %r9,%r2,0,1
  18:	bf 32 00 00 00 00 00 00 	mov %r2,%r3
  20:	bf 43 01 00 01 00 00 00 	addrc %r3,%r4,0,1
  28:	bf 65 01 00 00 00 01 00 	addrc %r5,%r6,1,0
