#as: -EB -mdialect=normal
#objdump: -dr -M dec
#source: jcond.s
#name: BPF conditional pseudo-jump instruction, normal syntax, big-endian

.*: +file format .*bpf.*

Disassembly of section .text:

0+ <.text>:
   0:	e5 00 00 00 00 00 00 00 	jcond 0
   8:	e5 00 00 01 00 00 00 00 	jcond 1
  10:	e5 00 ff fe 00 00 00 00 	jcond -2
  18:	e5 00 ff fd 00 00 00 00 	jcond -3
