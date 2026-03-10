# Source file used to test the MVI instruction.

foo:
	mvib	r24.w0, *r1.b2
	mvid	*r1.b0, *r1.b0
	mvid	*r1.b0, *--r1.b0
	mvid	*--r1.b0, *r1.b0++
	mviw	*r1.b0, *r1.b2++
	mviw	*r1.b1, *r1.b2++
	mvid	r20, *r1.b3++
	mvid	r20, *--r1.b2
	mvid	*--r1.b2, r31
