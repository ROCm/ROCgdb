	.text

	# An IFUNC with local binding: no .globl, so it has no entry in the
	# linker's hash table.  It is both called and referenced, so that
	# both its GOT entry and the data word holding its address need an
	# IRELATIVE.  The data word alone is the shape of glibc's configure
	# probe for linker IFUNC support.
	.type	local_ifunc, @gnu_indirect_function
local_ifunc:
	ret

	.globl	_start
	.ent	_start
_start:
	ldgp	$29, 0($27)
	ldq	$27, local_ifunc($29)	!literal!1
	jsr	$26, ($27), 0		!lituse_jsr!1
	ldgp	$29, 0($26)
	ret
	.end	_start

	.data
	.globl	ptr
ptr:
	.quad	local_ifunc
