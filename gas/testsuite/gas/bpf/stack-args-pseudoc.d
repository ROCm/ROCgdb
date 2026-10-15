#as: -EL -mdialect=pseudoc
#objdump: -dr -M dec,pseudoc
#source: stack-args-pseudoc.s
#name: BPF stack argument register, pseudoc syntax

.*: +file format .*bpf.*

Disassembly of section .text:

0+ <.text>:
   0:	79 b1 08 00 00 00 00 00 	r1=\*\(u64\*\)\(r11\+8\)
   8:	7b 2b f8 ff 00 00 00 00 	\*\(u64\*\)\(r11-8\)=r2
  10:	7a 0b f0 ff 04 00 00 00 	\*\(u64\*\)\(r11-16\)=4
