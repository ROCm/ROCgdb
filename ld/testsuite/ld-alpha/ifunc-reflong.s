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

	# An IRELATIVE writes the 64-bit address the resolver returns, so a
	# 32-bit reference to an IFUNC cannot be represented.  A local symbol
	# is named in the diagnostic by its symbol table entry rather than by
	# a hash table entry.
	.data
	.globl	ptr
ptr:
	.long	global_ifunc
	.long	local_ifunc
