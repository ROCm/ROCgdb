/* 8000 got entries against undefined symbols, so relaxation cannot remove
   any of them.  That is 64000 bytes; with the entry for the IFUNC it is
   just under MAX_GOT_SIZE, so this object keeps its own got subsection.  */

	.text
	.globl	global_ifunc
	.type	global_ifunc, @gnu_indirect_function
global_ifunc:
	ret

	.globl	_start
	.ent	_start
_start:
	ldgp	$29,0($27)
	.prologue 1
	ldq	$1,global_ifunc($29)	!literal
	.irpc	w,01234567
	.irpc	x,0123456789
	.irpc	y,0123456789
	.irpc	z,0123456789
	ldq	$1,g\w\x\y\z($29)	!literal
	.endr
	.endr
	.endr
	.endr
	ret	$31,($26),1
	.end	_start

	.data
	.globl	ptr
ptr:
	.quad	global_ifunc
