# Test TX49 extensions.
	.set	noreorder
	.set	noat
	.globl	text_label .text
text_label:
	madd	$1,$2
	madd	$3,$4,$5

	maddu	$6,$7
	maddu	$8,$9,$10

	mult	$11,$12,$13
	multu	$14,$15,$16

	dmult	$17,$18,$19
	dmultu	$20,$21,$22

	pref	4,($23)
	pref	4,32767($24)

	ctc0	$25,$26
	cfc0	$27,$28

	wait

	sdbbp
	sdbbp	0x56

	deret

# Force some (non-delay-slot) zero bytes, to make 'objdump' print ...
	.align	4, 0
	.space	16
