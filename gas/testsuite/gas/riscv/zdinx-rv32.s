target:
	fadd.d		a0, a2, a4
	fadd.d		a0, a2, a4, rne
	fsub.d		a0, a2, a4
	fsub.d		a0, a2, a4, rne
	fmul.d		a0, a2, a4
	fmul.d		a0, a2, a4, rne
	fdiv.d		a0, a2, a4
	fdiv.d		a0, a2, a4, rne
	fsqrt.d		a0, a2
	fsqrt.d		a0, a2, rne
	fmin.d		a0, a2, a4
	fmax.d		a0, a2, a4
	fmadd.d		a0, a2, a4, a6
	fmadd.d		a0, a2, a4, a6, rne
	fnmadd.d	a0, a2, a4, a6
	fnmadd.d	a0, a2, a4, a6, rne
	fmsub.d		a0, a2, a4, a6
	fmsub.d		a0, a2, a4, a6, rne
	fnmsub.d	a0, a2, a4, a6
	fnmsub.d	a0, a2, a4, a6, rne

	fcvt.w.d	a1, a2
	fcvt.w.d	a1, a2, rne
	fcvt.wu.d	a1, a2
	fcvt.wu.d	a1, a2, rne
	fcvt.d.w	a0, a1
	fcvt.d.wu	a0, a1

	fcvt.d.s	a0, a1
	fcvt.s.d	a1, a2
	fcvt.s.d	a1, a2, rne

	.option push
	.option arch, +zhinxmin
	fcvt.d.h	a0, a1
	fcvt.h.d	a1, a2
	fcvt.h.d	a1, a2, rne
	.option pop

	fsgnj.d		a0, a2, a4
	fsgnjn.d	a0, a2, a4
	fsgnjx.d	a0, a2, a4
	feq.d		a1, a2, a4
	flt.d		a1, a2, a4
	fle.d		a1, a2, a4
	fgt.d		a1, a2, a4
	fge.d		a1, a2, a4
	fmv.d		a0, a2
	fneg.d		a0, a2
	fabs.d		a0, a2
	fclass.d	a1, a2
