#as: -EB -mdialect=normal
#objdump: -dr -M dec
#source: acqrel.s
#name: BPF load-acquire and store-release instructions, normal syntax, big endian

.*: +file format .*bpf.*

Disassembly of section .text:

0+ <.text>:
   0:	d3 12 00 00 00 00 01 00 	ldacqb %r1,\[%r2\+0\]
   8:	cb 12 00 02 00 00 01 00 	ldacqh %r1,\[%r2\+2\]
  10:	c3 12 ff fc 00 00 01 00 	ldacqw %r1,\[%r2-4\]
  18:	db 12 7f ff 00 00 01 00 	ldacqdw %r1,\[%r2\+32767\]
  20:	d3 12 00 00 00 00 01 10 	strelb \[%r1\+0\],%r2
  28:	cb 12 00 02 00 00 01 10 	strelh \[%r1\+2\],%r2
  30:	c3 12 ff fc 00 00 01 10 	strelw \[%r1-4\],%r2
  38:	db 12 7f ff 00 00 01 10 	streldw \[%r1\+32767\],%r2
