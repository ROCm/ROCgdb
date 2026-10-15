#name: Prune JMP vectab, no code
#source: vectab-jmp.s
#source: prune-vtab-1.s
#target: avr-*-*
#as: -mavr51 -I$srcdir/$subdir
#ld: -mavr51 --relax --prune-vectab
#objdump: -d

.*:     file format elf32-avr


Disassembly of section .text:

00000000 <__vectors>:
   0:	f8 94       	cli

00000002 <__bad_interrupt>:
   2:	fe cf       	rjmp	.-4      	; 0x0 <__vectors>
