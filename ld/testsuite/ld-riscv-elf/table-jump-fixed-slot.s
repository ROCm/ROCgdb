	.text
	.globl _start
_start:
	.rept 5
	tail far_target
	.endr
	# These sites can shorten without a table.  Counting both targets
	# before shortening would need more than the script's single slot.
	.rept 5
	tail near_target
	.endr
near_target:
	ret

	.section .far,"ax",@progbits
far_target:
	ret
