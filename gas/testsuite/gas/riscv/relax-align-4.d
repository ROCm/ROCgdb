#as: -mrelax -march=rv64gc
#objdump: -dr

.*:[ ]+file format .*


Disassembly of section .text:

0+000 <b>:
[ 	]+0:[ 	]+0120006f[ 	]+j[ 	]+[0-9a-z]+[ 	]+<e>
[ 	]+0:[ 	]+R_RISCV_JAL[ 	]+e
[ 	]+4:[ 	]+0001[ 	]+\.insn[ 	]+2, 0x0001
[ 	]+4:[ 	]+R_RISCV_ALIGN[ 	]+\*ABS\*\+0x6
[ 	]+6:[ 	]+00000013[ 	]+nop

0+00a <c>:
[ 	]\.\.\.$
[ 	]+a:[ 	]+R_RISCV_64[ 	]+g

0+0012 <e>:
[ 	]+12:[ 	]+00013083[ 	]+ld[ 	]+ra,0\(sp\)
[ 	]+16:[ 	]+00000097[ 	]+auipc[ 	]+ra,0x0
[ 	]+16:[ 	]+R_RISCV_CALL_PLT[ 	]+h
[ 	]+1a:[ 	]+000080e7[ 	]+jalr[ 	]+ra[ 	]+#.*<e\+0x[0-9a-z]+>
[ 	]+1e:[ 	]+0001[ 	]+\.insn[ 	]+2, 0x0001
[ 	]+1e:[ 	]+R_RISCV_ALIGN[ 	]+\*ABS\*\+0x2

0+0020 <g>:
[ 	]+20:[ 	]+00000010[ 	]+\.word[ 	]+0x00000010
[ 	]+24:[ 	]+00000013[ 	]+nop

Disassembly of section .text.2:

0+000 <h>:
[ 	]+0:[ 	]+00000097[ 	]+auipc[ 	]+ra,0x0
[ 	]+0:[ 	]+R_RISCV_CALL_PLT[ 	]+b
[ 	]+0:[ 	]+R_RISCV_RELAX[ 	]+\*ABS\*
[ 	]+4:[ 	]+000080e7[ 	]+jalr[ 	]+ra[ 	]+#.*<h>
