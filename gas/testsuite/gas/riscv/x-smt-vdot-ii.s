target:
	# vmadot (xsmtvdotii only: i4 dtype)
	smt.vmadot v2, v3, v4, i4
	smt.vmadotu v2, v3, v4, i4
	smt.vmadotsu v2, v3, v4, i4
	smt.vmadotus v2, v3, v4, i4
	# vmadot default (i8)
	smt.vmadot v2, v3, v4
	smt.vmadotu v2, v3, v4
	smt.vmadotsu v2, v3, v4
	smt.vmadotus v2, v3, v4
	# vmadot explicit i8
	smt.vmadot v2, v3, v4, i8
	smt.vmadotu v2, v3, v4, i8
	smt.vmadotsu v2, v3, v4, i8
	smt.vmadotus v2, v3, v4, i8
	# vmadot1/2/3 (shared with xsmtvdot, i8 only)
	smt.vmadot1u v2, v4, v5
	smt.vmadot1 v2, v4, v5
	smt.vmadot1su v2, v4, v5
	smt.vmadot1us v2, v4, v5
	smt.vmadot2u v2, v4, v5
	smt.vmadot2 v2, v4, v5
	smt.vmadot2su v2, v4, v5
	smt.vmadot2us v2, v4, v5
	smt.vmadot3u v2, v4, v5
	smt.vmadot3 v2, v4, v5
	smt.vmadot3su v2, v4, v5
	smt.vmadot3us v2, v4, v5
	# vmadot.sp with Xp9 stride (0/1) and i4
	smt.vmadotu.sp v2, v4, v5, v0, 0, i4
	smt.vmadotu.sp v2, v4, v5, v0, 1, i4
	smt.vmadot.sp v2, v4, v5, v0, 0, i4
	smt.vmadotsu.sp v2, v4, v5, v0, 0, i4
	smt.vmadotus.sp v2, v4, v5, v0, 0, i4
	# vmadot.sp with Xp9 stride (0/1) and i8 (default)
	smt.vmadotu.sp v2, v4, v5, v0, 0, i8
	smt.vmadotu.sp v2, v4, v5, v0, 1, i8
	smt.vmadot.sp v2, v4, v5, v0, 0, i8
	smt.vmadotsu.sp v2, v4, v5, v0, 0, i8
	smt.vmadotus.sp v2, v4, v5, v0, 0, i8
	# vmadot.sp with Xpk stride (2/3) and i8
	smt.vmadotu.sp v2, v4, v5, v0, 2, i8
	smt.vmadotu.sp v2, v4, v5, v0, 3, i8
	smt.vmadot.sp v2, v4, v5, v0, 2, i8
	smt.vmadotsu.sp v2, v4, v5, v0, 2, i8
	smt.vmadotus.sp v2, v4, v5, v0, 2, i8
	# vmadot.sp with mask v1
	smt.vmadotu.sp v2, v4, v5, v1, 2, i8
	smt.vmadot.sp v2, v4, v5, v1, 2, i8
	# vmadot.hp
	smt.vmadotu.hp v2, v3, v4, v0, 0, i4
	smt.vmadotu.hp v2, v3, v4, v0, 1, i4
	smt.vmadotu.hp v2, v3, v4, v0, 0, i8
	smt.vmadot.hp v2, v3, v4, v0, 0, i4
	smt.vmadotsu.hp v2, v3, v4, v0, 0, i4
	smt.vmadotus.hp v2, v3, v4, v0, 0, i4
	smt.vmadotu.hp v2, v3, v4, v1, 0, i4
	# vfwmadot
	smt.vfwmadot v2, v3, v4
	smt.vfwmadot1 v2, v4, v5
	smt.vfwmadot2 v2, v4, v5
	smt.vfwmadot3 v2, v4, v5
	# vnpack/vnspack/vnpack4/vnspack4
	smt.vnpack.vv v2, v3, v4, 0
	smt.vnpack.vv v2, v3, v4, 1
	smt.vnspack.vv v2, v3, v4, 0
	smt.vnpack4.vv v2, v3, v4, 0
	smt.vnspack4.vv v2, v3, v4, 0
	# vpack/vupack
	smt.vpack.vv v2, v3, v4, 0
	smt.vpack.vv v2, v3, v4, 1
	smt.vupack.vv v2, v3, v4, 0
