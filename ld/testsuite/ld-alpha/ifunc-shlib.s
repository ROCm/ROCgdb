	.text

	# An IFUNC exported from a shared library.  A reference from another
	# module is resolved by the dynamic linker, not by the static linker,
	# so it is treated as an ordinary function here.
	.globl	shlib_ifunc
	.type	shlib_ifunc, @gnu_indirect_function
shlib_ifunc:
	ret
