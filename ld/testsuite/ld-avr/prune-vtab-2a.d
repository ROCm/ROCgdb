#name: Prune RJMP vectab, one ISR
#source: vectab-rjmp.s
#source: prune-vtab-2.s
#target: avr-*-*
#as: -mavr4 -I$srcdir/$subdir
#ld: -mavr4 --relax --prune-vectab
#objdump: -d

.*:     file format elf32-avr


Disassembly of section .text:

00000000 <__vectors>:
   0:	02 c0       	rjmp	.+4      	; 0x6 <__ctors_end>
   2:	02 c0       	rjmp	.+4      	; 0x8 <__bad_interrupt>
   4:	02 c0       	rjmp	.+4      	; 0xa <__vector_2>

00000006 <__ctors_end>:
   6:	f8 94       	cli

00000008 <__bad_interrupt>:
   8:	fb cf       	rjmp	.-10     	; 0x0 <__vectors>

0000000a <__vector_2>:
   a:	18 95       	reti
