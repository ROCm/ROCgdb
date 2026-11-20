#as: -mrelax
#objdump: -dr

.*:[ 	]+file format .*


Disassembly of section .text:

0+000 <rvc_func>:
[ 	]+0:[ 	]+8082[ 	]+ret
[ 	]+2:[ 	]+0001[ 	]+nop
[ 	]+4:[ 	]+0001[ 	]+nop
[ 	]+4:[ 	]+R_RISCV_ALIGN[ 	]+\*ABS\*\+0x2

0+006 <non_rvc_func>:
[ 	]+6:[ 	]+00008067[ 	]+ret

0+00a <insn>:
[ 	]+a:[ 	]+00000013[ 	]+nop
[ 	]+e:[ 	]+0000[ 	]+\.insn[ 	]+2, 0x0000
[ 	]+10:[ 	]+0001[ 	]+\.insn[ 	]+2, 0x0001
[ 	]+10:[ 	]+R_RISCV_ALIGN[ 	]+\*ABS\*\+0x2
[ 	]+12:[ 	]+00008067[ 	]+ret

0+0016 <hword>:
[ 	]+16:[ 	]+00000013[ 	]+nop
[ 	]+1a:[ 	]+0000[ 	]+\.short[ 	]+0x0000
[ 	]+1c:[ 	]+0001[ 	]+\.insn[ 	]+2, 0x0001
[ 	]+1c:[ 	]+R_RISCV_ALIGN[ 	]+\*ABS\*\+0x2
[ 	]+1e:[ 	]+00008067[ 	]+ret

0+0022 <byte>:
[ 	]+22:[ 	]+00000013[ 	]+nop
[ 	]+26:[ 	]+00[ 	]+\.byte[ 	]+0x00
[ 	]+27:[ 	]+00[ 	]+\.byte[ 	]+0x00
[ 	]+28:[ 	]+0001[ 	]+\.insn[ 	]+2, 0x0001
[ 	]+28:[ 	]+R_RISCV_ALIGN[ 	]+\*ABS\*\+0x2
[ 	]+2a:[ 	]+00008067[ 	]+ret
#pass
