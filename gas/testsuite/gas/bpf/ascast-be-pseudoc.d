#as: -EB -mdialect=pseudoc
#objdump: -dr -M dec,pseudoc
#source: ascast-pseudoc.s
#name: BPF address space cast instructions, pseudoc syntax, big endian

.*: +file format .*bpf.*

Disassembly of section .text:

0+ <.text>:
   0:	bf 11 00 01 00 00 00 01 	r1 = addr_space_cast\(r1, 0, 1\)
   8:	bf 01 00 01 00 01 00 00 	r0 = addr_space_cast\(r1, 1, 0\)
  10:	bf 92 00 01 00 00 00 01 	r9 = addr_space_cast\(r2, 0, 1\)
  18:	bf 23 00 00 00 00 00 00 	r2=r3
