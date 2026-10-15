#as: -EL -mdialect=normal
#objdump: -dr -M dec
#source: gotox.s
#name: BPF indirect jump instruction, normal syntax

.*: +file format .*bpf.*

Disassembly of section .text:

0+ <.text>:
   0:	0d 01 00 00 00 00 00 00 	ja %r1
   8:	0d 09 00 00 00 00 00 00 	ja %r9
  10:	05 00 01 00 00 00 00 00 	ja 1
  18:	95 00 00 00 00 00 00 00 	exit
