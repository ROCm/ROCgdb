#as: -EB -mdialect=normal
#objdump: -dr -M dec
#source: stack-args.s
#name: BPF stack argument register, normal syntax, big endian

.*: +file format .*bpf.*

Disassembly of section .text:

0+ <.text>:
   0:	79 1b 00 08 00 00 00 00 	ldxdw %r1,\[%r11\+8\]
   8:	7b b2 ff f8 00 00 00 00 	stxdw \[%r11-8\],%r2
  10:	7a b0 ff f0 00 00 00 04 	stdw \[%r11-16\],4
