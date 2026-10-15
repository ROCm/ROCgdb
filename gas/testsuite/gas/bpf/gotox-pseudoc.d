#as: -EL -mdialect=pseudoc
#objdump: -dr -M dec,pseudoc
#source: gotox-pseudoc.s
#name: BPF indirect jump instruction, pseudoc syntax

.*: +file format .*bpf.*

Disassembly of section .text:

0+ <.text>:
   0:	0d 01 00 00 00 00 00 00 	gotox r1
   8:	0d 09 00 00 00 00 00 00 	gotox r9
  10:	05 00 01 00 00 00 00 00 	goto 1
  18:	95 00 00 00 00 00 00 00 	exit
