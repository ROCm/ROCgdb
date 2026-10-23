	# The priv spec attributes are not generated even if the csr is used.
	.option arch, +zicsr
	csrr a0, 0x0
