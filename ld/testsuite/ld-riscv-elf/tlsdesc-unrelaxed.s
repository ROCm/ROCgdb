	.text
	.globl	_start
	.type	_start,@function
_start:
	/* The lo relocs come before the hi one, so the relax pass leaves
	   this sequence alone, and it is rewritten to IE in place.  */
	j	1f
2:
	addi	a0, a0, %tlsdesc_add_lo(.desc1)
	jalr	t0, t0, %tlsdesc_call(.desc1)
	add	a0, a0, tp
	ret
1:
.desc1:
	auipc	a0, %tlsdesc_hi(sg2)
	ld	t0, %tlsdesc_load_lo(.desc1)(a0)
	j	2b
