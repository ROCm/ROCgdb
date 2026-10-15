#as: -EB -mdialect=pseudoc
#objdump: -dr -M dec,pseudoc
#source: stack-args-pseudoc.s
#name: BPF stack argument register, pseudoc syntax, big endian

.*: +file format .*bpf.*

Disassembly of section .text:

0+ <.text>:
   0:	79 1b 00 08 00 00 00 00 	r1=\*\(u64\*\)\(r11\+8\)
   8:	7b b2 ff f8 00 00 00 00 	\*\(u64\*\)\(r11-8\)=r2
  10:	7a b0 ff f0 00 00 00 04 	\*\(u64\*\)\(r11-16\)=4
