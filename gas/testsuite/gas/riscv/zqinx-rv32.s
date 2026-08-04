target:
	fadd.q		s0, s4, s8
	fadd.q		s0, s4, s8, rne
	fsub.q		s0, s4, s8
	fsub.q		s0, s4, s8, rne
	fmul.q		s0, s4, s8
	fmul.q		s0, s4, s8, rne
	fdiv.q		s0, s4, s8
	fdiv.q		s0, s4, s8, rne
	fsqrt.q		s0, s4
	fsqrt.q		s0, s4, rne
	fmin.q		s0, s4, s8
	fmax.q		s0, s4, s8
	fmadd.q		s0, s4, s8, a2
	fmadd.q		s0, s4, s8, a2, rne
	fnmadd.q	s0, s4, s8, a2
	fnmadd.q	s0, s4, s8, a2, rne
	fmsub.q		s0, s4, s8, a2
	fmsub.q		s0, s4, s8, a2, rne
	fnmsub.q	s0, s4, s8, a2
	fnmsub.q	s0, s4, s8, a2, rne

	fcvt.w.q	a1, s4
	fcvt.w.q	a1, s4, rne
	fcvt.wu.q	a1, s4
	fcvt.wu.q	a1, s4, rne
	fcvt.q.w	s0, a1
	fcvt.q.wu	s0, a1

	fcvt.q.s	s0, a1
	fcvt.s.q	a1, s4
	fcvt.s.q	a1, s4, rne

	fcvt.q.d	s0, a0
	fcvt.d.q	a0, s4
	fcvt.d.q	a0, s4, rne

	.option push
	.option arch, +zhinxmin
	fcvt.q.h	s0, a1
	fcvt.h.q	a1, s4
	fcvt.h.q	a1, s4, rne
	.option pop

	fsgnj.q		s0, s4, s8
	fsgnjn.q	s0, s4, s8
	fsgnjx.q	s0, s4, s8
	feq.q		a1, s4, s8
	flt.q		a1, s4, s8
	fle.q		a1, s4, s8
	fgt.q		a1, s4, s8
	fge.q		a1, s4, s8

	fmv.q		s0, s4
