# Test DT_RELR with a GOT entry for a global symbol that is bound locally.

.option pic
.text
.global _start
_start:
	la	a0, _start
	ret
