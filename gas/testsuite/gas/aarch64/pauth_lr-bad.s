	/* FEAT_PAuth_LR Pointer Authentication instructions.  */
	.text

lr_signing:

	autiasppcr w0 // 32-bit registers not allowed
	autiasppcr sp // stack pointer register not allowed

	retaasppcr w0 // 32-bit registers not allowed
	retaasppcr sp // SP not allowed
	retaasppcr xzr // zero register not allowed
