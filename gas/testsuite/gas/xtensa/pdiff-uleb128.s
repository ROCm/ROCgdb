	.text
.L1:
	.space	200
.L2:

	.section	.debug_info, "", @progbits
	.uleb128	.L2 - .L1
