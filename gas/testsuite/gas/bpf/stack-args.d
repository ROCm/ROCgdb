#as: -EL -mdialect=normal
#objdump: -dr -M dec
#source: stack-args.s
#name: BPF stack argument register, normal syntax

.*: +file format .*bpf.*

Disassembly of section .text:

0+ <.text>:
   0:	79 b1 08 00 00 00 00 00 	ldxdw %r1,\[%r11\+8\]
   8:	7b 2b f8 ff 00 00 00 00 	stxdw \[%r11-8\],%r2
  10:	7a 0b f0 ff 04 00 00 00 	stdw \[%r11-16\],4
