/* Atomic 64-byte load/store consistency instructions.  */
.arch armv9-a+ls64+lsc64b

/* Single-copy Atomic 64-byte Load-acquire.  */
	lda64b x0, [x0]
	lda64b x0, [sp]
	lda64b x0, [x1, #0]
	lda64b x0, [x1]
	lda64b x14, [x1]
	lda64b x22, [x1]

/* Single-copy Atomic 64-byte Store-release without Status Result.  */
	stl64b x0, [x0]
	stl64b x0, [sp]
	stl64b x0, [x1, 0]
	stl64b x0, [x1]
	stl64b x14, [x1]
	stl64b x22, [x1]

/* Single-copy Atomic 64-byte Store-release with Status Result.  */
	stl64bv x0, x0, [x0]
	stl64bv xzr, x0, [x0]
	stl64bv x0, x0, [sp]
	stl64bv x1, x0, [x2]
	stl64bv x0, x14, [x2]
	stl64bv x0, x22, [x2]

/* Single-copy Atomic 64-byte EL0 Store-release with Status Result.  */
	stl64bv0 x0, x0, [x0]
	stl64bv0 xzr, x0, [x0]
	stl64bv0 x0, x0, [sp]
	stl64bv0 x1, x0, [x2]
	stl64bv0 x0, x14, [x2]
	stl64bv0 x0, x22, [x2]

/* Check that +lsc64b enables FEAT_LSCP without an explicit +lscp.  */
.arch armv9-a+lsc64b
	ldap xzr, x0, [x0]
	ldapp xzr, x0, [x0]
	stlp x0, x0, [x0]
