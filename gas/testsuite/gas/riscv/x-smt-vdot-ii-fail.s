target:
	# vmadot/vmadot1/2/3: vd must be even
	smt.vmadot v1, v3, v4
	smt.vmadotu v1, v3, v4
	smt.vmadotsu v1, v3, v4
	smt.vmadotus v1, v3, v4
	smt.vmadot1u v1, v4, v5
	smt.vmadot1 v1, v4, v5
	smt.vmadot1su v1, v4, v5
	smt.vmadot1us v1, v4, v5
	smt.vmadot2u v1, v4, v5
	smt.vmadot2 v1, v4, v5
	smt.vmadot2su v1, v4, v5
	smt.vmadot2us v1, v4, v5
	smt.vmadot3u v1, v4, v5
	smt.vmadot3 v1, v4, v5
	smt.vmadot3su v1, v4, v5
	smt.vmadot3us v1, v4, v5
	# vmadot1/2/3: vs1 must be even
	smt.vmadot1u v2, v3, v5
	smt.vmadot1 v2, v3, v5
	smt.vmadot1su v2, v3, v5
	smt.vmadot1us v2, v3, v5
	smt.vmadot2u v2, v3, v5
	smt.vmadot2 v2, v3, v5
	smt.vmadot2su v2, v3, v5
	smt.vmadot2us v2, v3, v5
	smt.vmadot3u v2, v3, v5
	smt.vmadot3 v2, v3, v5
	smt.vmadot3su v2, v3, v5
	smt.vmadot3us v2, v3, v5
	# vmadot1/2/3: i4 invalid (only i8 valid via Xpw)
	smt.vmadot1u v2, v4, v5, i4
	smt.vmadot1 v2, v4, v5, i4
	smt.vmadot1su v2, v4, v5, i4
	smt.vmadot1us v2, v4, v5, i4
	smt.vmadot2u v2, v4, v5, i4
	smt.vmadot2 v2, v4, v5, i4
	smt.vmadot2su v2, v4, v5, i4
	smt.vmadot2us v2, v4, v5, i4
	smt.vmadot3u v2, v4, v5, i4
	smt.vmadot3 v2, v4, v5, i4
	smt.vmadot3su v2, v4, v5, i4
	smt.vmadot3us v2, v4, v5, i4
	# vmadot/vmadot1/2/3: invalid data type
	smt.vmadot v2, v3, v4, i1
	smt.vmadot v2, v3, v4, i2
	smt.vmadot v2, v3, v4, i3
	smt.vmadot v2, v3, v4, i5
	smt.vmadot v2, v3, v4, i7
	smt.vmadot v2, v3, v4, i16
	smt.vmadot v2, v3, v4, i32
	smt.vmadot v2, v3, v4, i64
	smt.vmadot v2, v3, v4, i40
	smt.vmadot1u v2, v4, v5, i2
	smt.vmadot2u v2, v4, v5, i16
	smt.vmadot3u v2, v4, v5, i32

	# vmadot.sp: vd/vs1 must be even
	smt.vmadotu.sp v1, v4, v5, v0, 0, i8
	smt.vmadotu.sp v2, v3, v5, v0, 0, i8
	# vmadot.sp: mask must be v0/v1
	smt.vmadotu.sp v2, v4, v5, v2, 0, i8
	# vmadot.sp: i4 only valid with stride 0/1
	smt.vmadotu.sp v2, v4, v5, v0, 2, i4
	smt.vmadotu.sp v2, v4, v5, v0, 3, i4
	# vmadot.sp: stride >= 4 out of range
	smt.vmadotu.sp v2, v4, v5, v0, 4, i8
	# vmadot.hp: mask must be v0/v1
	smt.vmadotu.hp v2, v3, v4, v2, 0, i4
	# vmadot.hp: stride >= 8 out of range
	smt.vmadotu.hp v2, v3, v4, v0, 8, i4
	# vfwmadot: vd must be even
	smt.vfwmadot v1, v3, v4
	smt.vfwmadot1 v1, v4, v5
	smt.vfwmadot2 v1, v4, v5
	smt.vfwmadot3 v1, v4, v5
	# vfwmadot1/2/3: vs1 must be even
	smt.vfwmadot1 v2, v3, v5
	smt.vfwmadot2 v2, v3, v5
	smt.vfwmadot3 v2, v3, v5
	# vnpack/vnspack/vnpack4/vnspack4: imm2 out of range
	smt.vnpack.vv v2, v3, v4, 4
	smt.vnspack.vv v2, v3, v4, 4
	smt.vnpack4.vv v2, v3, v4, 4
	smt.vnspack4.vv v2, v3, v4, 4
	# vpack/vupack: vd must be even, imm2 out of range
	smt.vpack.vv v1, v3, v4, 0
	smt.vpack.vv v2, v3, v4, 4
	smt.vupack.vv v1, v3, v4, 0
	smt.vupack.vv v2, v3, v4, 4

	# xsmtvdotii - extension required (xsmtvdotii-only insns need xsmtvdotii)
	.option push
	.option arch, rv64gcv_xsmtvdot
	smt.vmadot v2, v3, v4, i4
	smt.vmadotu v2, v3, v4, i4
	smt.vmadotsu v2, v3, v4, i4
	smt.vmadotus v2, v3, v4, i4
	smt.vmadotu.sp v2, v4, v5, v0, 0, i4
	smt.vmadot.sp v2, v4, v5, v0, 0, i4
	smt.vmadotsu.sp v2, v4, v5, v0, 0, i4
	smt.vmadotus.sp v2, v4, v5, v0, 0, i4
	smt.vmadotu.hp v2, v3, v4, v0, 0, i4
	smt.vmadot.hp v2, v3, v4, v0, 0, i4
	smt.vmadotsu.hp v2, v3, v4, v0, 0, i4
	smt.vmadotus.hp v2, v3, v4, v0, 0, i4
	smt.vfwmadot v2, v3, v4
	smt.vfwmadot1 v2, v4, v5
	smt.vfwmadot2 v2, v4, v5
	smt.vfwmadot3 v2, v4, v5
	smt.vnpack.vv v2, v3, v4, 0
	smt.vnspack.vv v2, v3, v4, 0
	smt.vnpack4.vv v2, v3, v4, 0
	smt.vnspack4.vv v2, v3, v4, 0
	smt.vpack.vv v2, v3, v4, 0
	smt.vupack.vv v2, v3, v4, 0
	.option pop
