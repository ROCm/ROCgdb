#as: -EL -mdialect=pseudoc
#objdump: -dr -M dec,pseudoc
#source: jcond-pseudoc.s
#name: BPF conditional pseudo-jump instruction, pseudoc syntax

.*: +file format .*bpf.*

Disassembly of section .text:

0+ <.text>:
   0:	e5 00 00 00 00 00 00 00 	may_goto 0
   8:	e5 00 01 00 00 00 00 00 	may_goto 1
  10:	e5 00 fe ff 00 00 00 00 	may_goto -2
  18:	e5 00 fd ff 00 00 00 00 	may_goto -3
