#as: -EL -mdialect=pseudoc
#objdump: -dr -M dec,pseudoc
#source: ascast-pseudoc.s
#name: BPF address space cast instructions, pseudoc syntax

.*: +file format .*bpf.*

Disassembly of section .text:

0+ <.text>:
   0:	bf 11 01 00 01 00 00 00 	r1 = addr_space_cast\(r1, 0, 1\)
   8:	bf 10 01 00 00 00 01 00 	r0 = addr_space_cast\(r1, 1, 0\)
  10:	bf 29 01 00 01 00 00 00 	r9 = addr_space_cast\(r2, 0, 1\)
  18:	bf 32 00 00 00 00 00 00 	r2=r3
  20:	bf 43 01 00 01 00 00 00 	r3 = addr_space_cast\(r4, 0, 1\)
  28:	bf 65 01 00 00 00 01 00 	r5 = addr_space_cast\(r6, 1, 0\)
