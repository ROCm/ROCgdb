#name: Prune JMP+CLH vectab, no code
#source: vectab-jmp-clh.s
#source: prune-vtab-1.s
#target: avr-*-*
#as: -mavr51 -I$srcdir/$subdir
#ld: -mavr51 --relax --prune-vectab
#objdump: -d

.*:     file format elf32-avr


Disassembly of section .text:

00000000 <__vectors>:
   0:	01 c0       	rjmp	.+2      	; 0x4 <__ctors_end>
   2:	d8 94       	clh

00000004 <__ctors_end>:
   4:	f8 94       	cli

00000006 <__bad_interrupt>:
   6:	fc cf       	rjmp	.-8      	; 0x0 <__vectors>
