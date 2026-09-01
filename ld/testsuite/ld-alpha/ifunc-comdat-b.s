	# This copy of the group is discarded in favour of the one in
	# ifunc-comdat-a.s, taking the definition of local_ifunc with it.
	.section .text.grp,"axG",@progbits,grp,comdat
	.type	local_ifunc, @gnu_indirect_function
local_ifunc:
	ret

	# The reference survives, but nothing relocates it: the generic code
	# reports it rather than letting relocate_section reach it.  Nothing
	# may be reserved in .rela.iplt for it either.
	.data
	.globl	ptr
ptr:
	.quad	local_ifunc
