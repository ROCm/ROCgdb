#as: -EB -mdialect=pseudoc
#objdump: -dr -M pseudoc
#source: hmul-pseudoc.s
#name: BPF UHMUL and SHMUL insns, big endian, pseudo-c syntax

.*: +file format .*bpf.*

Disassembly of section .text:

0+ <.text>:
   0:	2f 21 00 01 00 00 00 00 	r2 uh\*=r1
   8:	27 20 00 01 00 00 00 21 	r2 uh\*=33
  10:	27 20 00 01 ff ff ff f9 	r2 uh\*=-7
  18:	2f 52 00 02 00 00 00 00 	r5 sh\*=r2
  20:	27 50 00 02 00 00 00 09 	r5 sh\*=9
  28:	27 50 00 02 ff ff ff fd 	r5 sh\*=-3
