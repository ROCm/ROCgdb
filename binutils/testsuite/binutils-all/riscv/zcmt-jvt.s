	# Construct the instructions and table directly.  Linking only resolves
	# the table addresses; no linker relaxation is needed by this test.
	.option norelax
	.text
	.globl foo, bar
foo:
	cm.jt 0
bar:
	cm.jalt 32

	.section .riscv.jvt, "ax"
	# The linker script places the table at a 64-byte-aligned address.
	.ifdef with_base
	.globl __jvt_base$
__jvt_base$:
	.endif
	.if entry_size == 8
	.dword foo
	.rept 31
	.dword 0
	.endr
	.dword bar
	.else
	.word foo
	.rept 31
	.word 0
	.endr
	.word bar
	.endif
