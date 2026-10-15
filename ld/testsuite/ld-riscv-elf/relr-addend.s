# Test that the link-time address is written in place for packed
# relocations, since DT_RELR has no addend field.

.option pic
.data
.p2align 3
x:
	.quad	0x114514
y:
	.quad	0x1919810
px:
	.quad	x
py:
	.quad	y

.text
.global _start
_start:
	la	a0, x
	la	a1, y
