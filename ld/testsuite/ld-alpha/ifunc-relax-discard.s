/* As ifunc-relax-a.s, but with this object's reference to the IFUNC in a
   section that the linker script discards.  */

	.text
	.globl	global_ifunc
	.type	global_ifunc, @gnu_indirect_function
global_ifunc:
	ret

	# The only reference in this object to the IFUNC's got entry, in a
	# section the linker script discards.  The entry is still created and
	# still occupies the got, so it is the one that survives when the two
	# got subsections merge, and it has to take the mark from the entry
	# that the surviving reference in ifunc-relax-b.s made.
	.section .text.discard,"ax",@progbits
	ldgp	$29,0($27)
	ldq	$1,global_ifunc($29)	!literal
	ret

	.text
	.globl	_start
	.ent	_start
_start:
	ldgp	$29,0($27)
	.prologue 1
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
