	.ifdef __64_bit__
	.equ ALIGN, 3
	.else
	.equ ALIGN, 2
	.endif

	.section ".note.gnu.property", "a"
	.p2align ALIGN
	.long 1f - 0f		/* name length.  */
	.long 3f - 1f		/* data length.  */
	/* NT_GNU_PROPERTY_TYPE_0 */
	.long 5			/* note type.  */
0:
	.asciz "GNU"		/* vendor name.  */
1:
	.p2align ALIGN
	/* GNU_PROPERTY_X86_ISA_1_USED */
	.long 0xc0010002	/* pr_type.  */
	.long 5f - 4f		/* pr_datasz.  */
4:
	.long 0x3 /* GNU_PROPERTY_X86_ISA_1_BASELINE | GNU_PROPERTY_X86_ISA_1_V2 */
5:
	.p2align ALIGN
	/* GNU_PROPERTY_X86_ISA_1_NEEDED */
	.long 0xc0008002	/* pr_type.  */
	.long 5f - 4f		/* pr_datasz.  */
4:
	.long 0xa /* GNU_PROPERTY_X86_ISA_1_V2 | GNU_PROPERTY_X86_ISA_1_V4 */
5:
	.p2align ALIGN
3:
	.section	.note.GNU-stack
