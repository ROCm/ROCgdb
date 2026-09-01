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

	# The IRELATIVE for each of these is applied to a read-only page:
	# DT_TEXTREL in a dynamic link, an error in a static one.
	.section .rodata,"a",@progbits
	.quad	global_ifunc
	.quad	local_ifunc
