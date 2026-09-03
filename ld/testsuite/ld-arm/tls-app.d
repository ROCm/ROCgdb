
.*:     file format elf32-.*arm.*
architecture: arm.*, flags 0x00000112:
EXEC_P, HAS_SYMS, D_PAGED
start address 0x00008[0-9a-f]+

Disassembly of section .text:

00008[0-9a-f]+ <foo>:
    8[0-9a-f]+:	e1a00000 	nop			@ \(mov r0, r0\)
    8[0-9a-f]+:	e1a00000 	nop			@ \(mov r0, r0\)
    8[0-9a-f]+:	e1a0f00e 	mov	pc, lr
    8[0-9a-f]+:	000080(bc|c4) .*
    8[0-9a-f]+:	000080(b4|bc) .*
    8[0-9a-f]+:	000080(ac|b4) .*
    8[0-9a-f]+:	00000004 .*
    8[0-9a-f]+:	000080(c4|cc) .*
    8[0-9a-f]+:	00000014 .*
