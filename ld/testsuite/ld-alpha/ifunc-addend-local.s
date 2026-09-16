	.text

	.type	local_ifunc, @gnu_indirect_function
local_ifunc:
	ret
	ret

	# A local symbol is not preemptible, so even in a shared library the
	# reference becomes an R_ALPHA_IRELATIVE and cannot carry an offset.
	.data
	.globl	ptr
ptr:
	.quad	local_ifunc+4
