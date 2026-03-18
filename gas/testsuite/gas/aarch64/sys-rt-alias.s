// sys-rt-alias.s Test file for AArch64 instructions where Rt !=31 is undefined behaviour.

	.text
	sys #4, c8, c7, #4      // TLBI ALLE1 with Rt=31
	sys #4, c8, c7, #4, x0  // TLBI ALLE1 with Rt!=31
	sys #4, c10, c7, #4     // PLBI ALLE1 with Rt=31
	sys #4, c10, c7, #4, x0 // PLBI ALLE1 with Rt!=31
	sys #4, c7, c0, #4      // MLBI ALLE1 with Rt=31
	sys #4, c7, c0, #4, x0  // MLBI ALLE1 with Rt!=31
	sys #0, c7, c5, #0      // IC IALLU with Rt=31
	sys #0, c7, c5, #0, x0  // IC IALLU with Rt!=31
	sys #0, c8, c7, #5, xzr // TLBI VALE1 with Rt=31 (valid Xt form)
	sys #0, c8, c7, #5, x0  // TLBI VALE1 with Rt!=31 (valid Xt form)
