/* Atomic 64-byte load/store consistency instructions require Rt to be even
   and <= 22.  */
.arch armv9-a+ls64+lsc64b

/* Single-copy Atomic 64-byte Load-acquire.  */
	lda64b x1, [x1]
	lda64b x13, [x1]
	lda64b x21, [x1]
	lda64b x23, [x1]
	lda64b x24, [x1]
	lda64b x30, [x1]
	lda64b xzr, [x1]

/* Single-copy Atomic 64-byte Store-release without Status Result.  */
	stl64b x1, [x1]
	stl64b x13, [x1]
	stl64b x21, [x1]
	stl64b x23, [x1]
	stl64b x24, [x1]
	stl64b x30, [x1]
	stl64b xzr, [x1]


/* Single-copy Atomic 64-byte Store-release with Status Result.  */
	stl64bv x0, x1,  [x2]
	stl64bv x0, x13, [x1]
	stl64bv x0, x21, [x1]
	stl64bv x0, x23, [x1]
	stl64bv x0, x24, [x1]
	stl64bv x0, x30, [x1]
	stl64bv x0, xzr, [x1]


/* Single-copy Atomic 64-byte EL0 Store-release with Status Result.  */
	stl64bv0 x0, x1,  [x2]
	stl64bv0 x0, x13, [x1]
	stl64bv0 x0, x21, [x1]
	stl64bv0 x0, x23, [x1]
	stl64bv0 x0, x24, [x1]
	stl64bv0 x0, x30, [x1]
	stl64bv0 x0, xzr, [x1]

/* Omit +ls64 in .arch.  */
.arch armv9-a+lsc64b
	lda64b x0, [x1]
	stl64b x0, [x1]
	stl64bv x1, x0, [x2]
	stl64bv0 x1, x0, [x2]
