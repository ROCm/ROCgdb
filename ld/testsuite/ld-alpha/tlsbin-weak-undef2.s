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
	ldah $0,x($0)		!tprelhi
	ldl $1,x($0)		!tprello
	ret
	.end _start
	.weak x
	.hidden x
	.section	.note.GNU-stack,"",@progbits
