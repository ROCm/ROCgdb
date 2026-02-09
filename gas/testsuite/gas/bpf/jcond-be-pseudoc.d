#as: -EB -mdialect=pseudoc
#objdump: -dr -M dec,pseudoc
#source: jcond-pseudoc.s
#name: BPF conditional pseudo-jump instruction, pseudoc syntax, big-endian

.*: +file format .*bpf.*

Disassembly of section .text:

0+ <.text>:
   0:	e5 00 00 00 00 00 00 00 	may_goto 0
   8:	e5 00 00 01 00 00 00 00 	may_goto 1
  10:	e5 00 ff fe 00 00 00 00 	may_goto -2
  18:	e5 00 ff fd 00 00 00 00 	may_goto -3
