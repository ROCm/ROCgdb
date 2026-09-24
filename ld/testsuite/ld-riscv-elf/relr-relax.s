# Test DT_RELR when relaxation deletes bytes before packed relocations.

.text
.p2align 3
.global _start
_start:
	call	foo
	call	foo
.p2align 3
x:
	.quad	foo
	.quad	foo
foo:
	ret
