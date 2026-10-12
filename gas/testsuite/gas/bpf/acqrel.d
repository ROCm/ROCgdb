#as: -EL -mdialect=normal
#objdump: -dr -M dec
#source: acqrel.s
#name: BPF load-acquire and store-release instructions, normal syntax

.*: +file format .*bpf.*

Disassembly of section .text:

0+ <.text>:
   0:	d3 21 00 00 00 01 00 00 	ldacqb %r1,\[%r2\+0\]
   8:	cb 21 02 00 00 01 00 00 	ldacqh %r1,\[%r2\+2\]
  10:	c3 21 fc ff 00 01 00 00 	ldacqw %r1,\[%r2-4\]
  18:	db 21 ff 7f 00 01 00 00 	ldacqdw %r1,\[%r2\+32767\]
  20:	d3 21 00 00 10 01 00 00 	strelb \[%r1\+0\],%r2
  28:	cb 21 02 00 10 01 00 00 	strelh \[%r1\+2\],%r2
  30:	c3 21 fc ff 10 01 00 00 	strelw \[%r1-4\],%r2
  38:	db 21 ff 7f 10 01 00 00 	streldw \[%r1\+32767\],%r2
