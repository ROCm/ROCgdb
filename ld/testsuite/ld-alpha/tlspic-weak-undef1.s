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
	ldq $1,x($29)		!gotdtprel
	addq $0,$1,$1
	ret
	.end _start
	.section .tbss,"awT",@nobits
	.type	y, @object
	.size	y, 4
y:
	.zero	4
	.weak x
	.hidden x
	.section	.note.GNU-stack,"",@progbits
