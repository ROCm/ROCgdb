#name: Prune JMP vectab, one ISR
#source: vectab-jmp.s
#source: prune-vtab-2.s
#target: avr-*-*
#as: -mavr51 -I$srcdir/$subdir
#ld: -mavr51 --relax --prune-vectab
#objdump: -d

.*:     file format elf32-avr


Disassembly of section .text:

00000000 <__vectors>:
   0:	04 c0       	rjmp	.+8      	; 0xa <__ctors_end>
   2:	00 00       	nop
   4:	03 c0       	rjmp	.+6      	; 0xc <__bad_interrupt>
   6:	00 00       	nop
   8:	02 c0       	rjmp	.+4      	; 0xe <__vector_2>

0000000a <__ctors_end>:
   a:	f8 94       	cli

0000000c <__bad_interrupt>:
   c:	f9 cf       	rjmp	.-14     	; 0x0 <__vectors>

0000000e <__vector_2>:
   e:	18 95       	reti
