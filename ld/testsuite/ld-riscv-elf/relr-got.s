# Test GOT relocations with DT_RELR.

.text
.option norelax
.option pic
.global _start
_start:

sym_local:
	nop

.global sym_hidden
.hidden sym_hidden
sym_hidden:
	nop

.global sym_global
sym_global:
	nop

.global sym_global_abs
.set sym_global_abs, 42

.global sym_weak_undef
.weak sym_weak_undef

.section .text.got_local, "ax"
	la      x1, sym_local

.section .text.got_hidden, "ax"
	la      x1, sym_hidden

.section .text.got_global, "ax"
	la      x1, sym_global

.section .text.got_global_abs, "ax"
	la	x1, sym_global_abs

.section .text.got_weak_undef, "ax"
	la      x1, sym_weak_undef

.section .text.got_DYNAMIC, "ax"
	la      x1, _DYNAMIC
