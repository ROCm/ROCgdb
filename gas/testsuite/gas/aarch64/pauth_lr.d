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
