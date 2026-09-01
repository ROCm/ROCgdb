	.text

	# A personality routine named with DW_EH_PE_absptr puts a REFQUAD
	# against the IFUNC in .eh_frame.  Both objects name the same one, so
	# their CIEs are identical and .eh_frame editing deletes one of them,
	# along with the place of the relocation that check_relocs counted.
	.globl	global_ifunc
	.type	global_ifunc, @gnu_indirect_function
global_ifunc:
	ret

	.globl	_start
	.ent	_start
_start:
	.cfi_startproc
	.cfi_personality 0x00, global_ifunc
	ret
	.cfi_endproc
	.end	_start
