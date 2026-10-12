	.section	.text.dead,"axG",@progbits,dead,comdat
	.global	dead
dead:
	ret

	.text
	.global	_start
_start:
	ret
