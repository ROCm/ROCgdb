#as: -EL -mdialect=normal
#objdump: -dr -M dec
#source: jcond.s
#name: BPF conditional pseudo-jump instruction, normal syntax, little-endian

.*: +file format .*bpf.*

Disassembly of section .text:

0+ <.text>:
   0:	e5 00 00 00 00 00 00 00 	jcond 0
   8:	e5 00 01 00 00 00 00 00 	jcond 1
  10:	e5 00 fe ff 00 00 00 00 	jcond -2
  18:	e5 00 fd ff 00 00 00 00 	jcond -3
