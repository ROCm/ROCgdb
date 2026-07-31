target:
	fadd.q		s1, s4, s8
	fadd.q		s0, s5, s8
	fadd.q		s0, s4, s9
	fsub.q		s1, s4, s8
	fsub.q		s0, s5, s8
	fsub.q		s0, s4, s9
	fmul.q		s1, s4, s8
	fmul.q		s0, s5, s8
	fmul.q		s0, s4, s9
	fdiv.q		s1, s4, s8
	fdiv.q		s0, s5, s8
	fdiv.q		s0, s4, s9
	fsqrt.q		s1, s4
	fsqrt.q		s0, s5
	fmin.q		s1, s4, s8
	fmin.q		s0, s5, s8
	fmin.q		s0, s4, s9
	fmax.q		s1, s4, s8
	fmax.q		s0, s5, s8
	fmax.q		s0, s4, s9
	fmadd.q		s1, s4, s8, a2
	fmadd.q		s0, s5, s8, a2
	fmadd.q		s0, s4, s9, a2
	fmadd.q		s0, s4, s8, a3
	fnmadd.q	s1, s4, s8, a2
	fnmadd.q	s0, s5, s8, a2
	fnmadd.q	s0, s4, s9, a2
	fnmadd.q	s0, s4, s8, a3
	fmsub.q		s1, s4, s8, a2
	fmsub.q		s0, s5, s8, a2
	fmsub.q		s0, s4, s9, a2
	fmsub.q		s0, s4, s8, a3
	fnmsub.q	s1, s4, s8, a2
	fnmsub.q	s0, s5, s8, a2
	fnmsub.q	s0, s4, s9, a2
	fnmsub.q	s0, s4, s8, a3

	fcvt.w.q	a0, s5
	fcvt.wu.q	a0, s5
	fcvt.q.w	s1, a0
	fcvt.q.wu	s1, a0

	fcvt.q.s	s1, a0
	fcvt.s.q	a0, s5

	fcvt.q.d	s1, a0
	fcvt.d.q	a0, s5

	.option push
	.option arch, +zhinxmin
	fcvt.q.h	s1, a0
	fcvt.h.q	a0, s5
	.option pop

	fsgnj.q		s1, s4, s8
	fsgnjn.q	s0, s5, s8
	fsgnjx.q	s0, s4, s9
	feq.q		a0, s5, s8
	flt.q		a0, s4, s9
	fle.q		a0, s5, s8
	fgt.q		a0, s4, s9
	fge.q		a0, s5, s8
	fmv.q		s1, s4
	fmv.q		s0, s5
	fneg.q		s1, s4
	fneg.q		s0, s5
	fabs.q		s1, s4
	fabs.q		s0, s5
	fclass.q	a0, s5
