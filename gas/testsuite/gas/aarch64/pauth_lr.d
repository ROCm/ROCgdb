#source: pauth_lr.s
#as: -march=armv9.5-a+pauth-lr
#objdump: -dr

.*:     file .*

Disassembly of section \.text:

0+ <.*>:
.*:	d50324ff 	pacm
.*:	dac1a3fe 	paciasppc
.*:	dac1a7fe 	pacibsppc
.*:	dac18bfe 	pacia171615
.*:	dac18ffe 	pacib171615
.*:	dac183fe 	pacnbiasppc
.*:	dac187fe 	pacnbibsppc
.*:	dac1bbfe 	autia171615
.*:	dac1bffe 	autib171615
.*:	dac1901e 	autiasppcr	x0
.*:	dac193fe 	autiasppcr	xzr
.*:	dac193de 	autiasppcr	x30
.*:	dac1941e 	autibsppcr	x0
.*:	d65f0be0 	retaasppcr	x0
.*:	d65f0bfe 	retaasppcr	x30
.*:	d65f0fe0 	retabsppcr	x0
[^\s]+\s<lr_signing>:
      40:	f380001f 	autiasppc	40 <lr_signing>
      44:	f3a0003f 	autibsppc	40 <lr_signing>
      48:	5500005f 	retaasppc	40 <lr_signing>
      4c:	5520007f 	retabsppc	40 <lr_signing>
      50:	5500001f 	retaasppc	50 <lr_signing\+0x10>
      54:	5500001f 	retaasppc	54 <lr_signing\+0x14>
      58:	5500003f 	retaasppc	54 <lr_signing\+0x14>
      5c:	5500011f 	retaasppc	3c <lr_signing-0x4>
      60:	550002df 	retaasppc	8 <lr_signing-0x38>
#...
   40000:	55091a7f 	retaasppc	2dcb4 <lr_signing\+0x2dc74>
   40004:	551fffff 	retaasppc	8 <lr_signing-0x38>
