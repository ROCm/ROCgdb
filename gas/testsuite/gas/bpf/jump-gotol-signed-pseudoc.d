#as: -EL -mdialect=pseudoc
#objdump: -dr -M dec,pseudoc
#source: jump-gotol-signed-pseudoc.s
#name: eBPF gotol with signed offsets, pseudoc syntax

.*: +file format .*bpf.*

Disassembly of section .text:

0+ <.text>:
   0:	06 00 00 00 01 00 00 00 	gotol 1
   8:	06 00 00 00 ff ff ff ff 	gotol -1
  10:	06 00 00 00 01 00 00 00 	gotol 1
  18:	06 00 00 00 00 00 00 00 	gotol 0
  20:	06 00 00 00 00 00 00 00 	gotol 0
