#objdump: -dr

.*:     file format .*

Disassembly of section \.text:

0+ <.*>:
.*:	d50c879f 	tlbi	alle1
.*:	d50c8780 	sys	#4, C8, C7, #4, x0
.*:	d50ca79f 	plbi	alle1
.*:	d50ca780 	sys	#4, C10, C7, #4, x0
.*:	d50c709f 	mlbi	alle1
.*:	d50c7080 	sys	#4, C7, C0, #4, x0
.*:	d508751f 	ic	iallu
.*:	d5087500 	sys	#0, C7, C5, #0, x0
.*:	d50887bf 	tlbi	vale1, xzr
.*:	d50887a0 	tlbi	vale1, x0
