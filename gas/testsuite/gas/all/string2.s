	.data

	.if 077 == 77
	.ascii <33D>
	.ascii <3fH>
	.ascii <134O>
	.ascii <100001B>
	.ascii <33D>,< 3fH ><134O> <100001B>
	.else
	.ascii <33>
	.ascii <0x3f>
	.ascii <0134>
	.ascii <0b100001>
	.ascii <33>,< 0x3f ><0134> <0b100001>
	.endif
