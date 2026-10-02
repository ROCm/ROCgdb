#name: Prune JMP+CLH vectab, one ISR
#source: vectab-jmp-clh.s
#source: prune-vtab-2.s
#target: avr-*-*
#as: -mavr51 -I$srcdir/$subdir
#ld: -mavr51 --relax --prune-vectab
#objdump: -d

.*:     file format elf32-avr


Disassembly of section .text:

00000000 <__vectors>:
   0:	05 c0       	rjmp	.+10     	; 0xc <__ctors_end>
   2:	00 00       	nop
   4:	04 c0       	rjmp	.+8      	; 0xe <__bad_interrupt>
   6:	00 00       	nop
   8:	03 c0       	rjmp	.+6      	; 0x10 <__vector_2>
   a:	d8 94       	clh

0000000c <__ctors_end>:
   c:	f8 94       	cli

0000000e <__bad_interrupt>:
   e:	f8 cf       	rjmp	.-16     	; 0x0 <__vectors>

00000010 <__vector_2>:
  10:	18 95       	reti
