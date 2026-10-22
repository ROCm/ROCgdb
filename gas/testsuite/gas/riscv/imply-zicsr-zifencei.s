target:
	# zicsr
	csrr	t0, mscratch
	csrwi	mscratch, 0x0
	csrsi	mscratch, 0x0
	csrci	mscratch, 0x0
	csrw	mscratch, t0
	csrw	mscratch, 0x0
	csrs	mscratch, t0
	csrs	mscratch, 0x0
	csrc	mscratch, t0
	csrc	mscratch, 0x0
	csrrwi	t0, mscratch, 0x0
	csrrsi	t0, mscratch, 0x0
	csrrci	t0, mscratch, 0x0
	csrrw	t0, mscratch, t0
	csrrw	t0, mscratch, 0x0
	csrrs	t0, mscratch, t0
	csrrs	t0, mscratch, 0x0
	csrrc	t0, mscratch, t0
	csrrc	t0, mscratch, 0x0

	# zifencei
	fence.i
