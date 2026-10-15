	.text
	.globl	_start
	.type	_start,@function
_start:
	/* TLSDESC to a var in a shared library, relaxed to IE (RV32).  */
.desc1:
	auipc	a0, %tlsdesc_hi(sg2)
	lw	t0, %tlsdesc_load_lo(.desc1)(a0)
	addi	a0, a0, %tlsdesc_add_lo(.desc1)
	jalr	t0, t0, %tlsdesc_call(.desc1)
	add	a0, a0, tp
	ret
