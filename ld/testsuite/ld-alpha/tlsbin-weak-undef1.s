	.set noreorder
	.set volatile
	.set noat
	.set nomacro
	.arch ev4
	.text
	.align 4
	.globl _start
	.ent _start
_start:
	ldq $1,x($29)		!gottprel
	addq $0,$1,$0
	ret
	.end _start
	.weak x
	.hidden x
	.section	.note.GNU-stack,"",@progbits
