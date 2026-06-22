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
	# vmadot/vmadot1/2/3: invalid data type
	smt.vmadot v2, v3, v4, i1
	smt.vmadot v2, v3, v4, i2
	smt.vmadot v2, v3, v4, i3
	smt.vmadot v2, v3, v4, i4
	smt.vmadot v2, v3, v4, i5
	smt.vmadot v2, v3, v4, i7
	smt.vmadot v2, v3, v4, i16
	smt.vmadot v2, v3, v4, i32
	smt.vmadot v2, v3, v4, i64
	smt.vmadot1u v2, v4, v5, i2
	smt.vmadot2u v2, v4, v5, i4
	smt.vmadot3u v2, v4, v5, i16

	# xsmtvdot - extension required
	.option push
	.option arch, rv64gcv
	.include "x-smt-vdot.s"
	.option pop
