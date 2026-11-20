	.altmacro
	.section .text

	# Test JVT overflow: cm.jalt has 224 entries (index 32-255)
	# func_top: called 10x (highest benefit), guaranteed jvt.jalt[32]
	# func_0 to func_222: called 2x each, fill remaining 223 JVT slots
	# func_overflow: called 1x (lowest benefit), overflows, remains jal

	.macro call2 n
	call func_\n
	call func_\n
	.endm

	.macro defn n
	.type func_\n, %function
func_\n:	ret
	.endm

	.globl _start
	.type	_start, %function
_start:
	# func_top: highest benefit, guaranteed first JVT entry
	.rept 10
	call func_top
	.endr

	# 223 functions with 2 calls each (fill remaining JVT entries)
	.set i, 0
	.rept 223
	call2 %i
	.set i, i+1
	.endr

	# func_overflow: lowest benefit, will not fit in JVT
	call func_overflow
	ret

	# Define functions
	.type func_top, %function
func_top:	ret

	.set i, 0
	.rept 223
	defn %i
	.set i, i+1
	.endr

	.type func_overflow, %function
func_overflow:	ret
