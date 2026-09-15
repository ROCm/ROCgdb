	.text

	.globl	global_ifunc
	.type	global_ifunc, @gnu_indirect_function
global_ifunc:
	ret

	.type	local_ifunc, @gnu_indirect_function
local_ifunc:
	ret

	.globl	_start
	.ent	_start
_start:
	ret
	.end	_start

	# The one reference that survives, so that .rela.iplt is not simply
	# empty.
	.data
	.globl	ptr
ptr:
	.quad	global_ifunc

	# A linker script discards these after check_relocs has seen them.
	# Nothing relocates them, so nothing may be reserved for them.
	.section .data.discard,"aw",@progbits
	.quad	global_ifunc
	.quad	local_ifunc

	.section .text.discard,"ax",@progbits
	ldq	$27, global_ifunc($29)	!literal!1
	ldq	$27, local_ifunc($29)	!literal!2
