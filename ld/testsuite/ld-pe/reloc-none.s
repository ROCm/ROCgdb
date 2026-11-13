	.text
	.global _start
_start:
	.nop

	.data
	.reloc ., BFD_RELOC_NONE, _start
	.long 0
	.reloc ., BFD_RELOC_NONE, sym
	.long 0
	.reloc ., BFD_RELOC_NONE, .
	.long 0
