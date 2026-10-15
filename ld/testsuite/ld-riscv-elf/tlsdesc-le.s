	.section	.tbss,"awT",@nobits
	/* Make the TP offset of sl need both lui and addi.  */
	.zero	0x1234
	.globl	sl
	.type	sl,@object
	.size	sl,4
sl:
	.zero	4

	.text
	.globl	_start
	.type	_start,@function
_start:
	/* TLSDESC to a var in the executable, relaxed to LE.  */
.desc1:
	auipc	a0, %tlsdesc_hi(sl)
	ld	t0, %tlsdesc_load_lo(.desc1)(a0)
	addi	a0, a0, %tlsdesc_add_lo(.desc1)
	jalr	t0, t0, %tlsdesc_call(.desc1)
	add	a0, a0, tp
	ret
