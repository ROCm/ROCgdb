#as: -EL -mdialect=pseudoc
#objdump: -dr -M pseudoc
#source: hmul-pseudoc.s
#name: BPF UHMUL and SHMUL insns, pseudo-c syntax

.*: +file format .*bpf.*

Disassembly of section .text:

0+ <.text>:
   0:	2f 12 01 00 00 00 00 00 	r2 uh\*=r1
   8:	27 02 01 00 21 00 00 00 	r2 uh\*=33
  10:	27 02 01 00 f9 ff ff ff 	r2 uh\*=-7
  18:	2f 25 02 00 00 00 00 00 	r5 sh\*=r2
  20:	27 05 02 00 09 00 00 00 	r5 sh\*=9
  28:	27 05 02 00 fd ff ff ff 	r5 sh\*=-3
