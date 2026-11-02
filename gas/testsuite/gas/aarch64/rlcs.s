/* Release Consistency Scoping instructions.  */
.arch armv9-a

/* Scoped Release Boundary.  */
	slbnd

/* Scope Release.  */
	srls

/* Scope Release Store Shared Stream.  */
	srls stshstrm
