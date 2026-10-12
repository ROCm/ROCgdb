	/* FEAT_PAuth_LR Pointer Authentication instructions.  */
	.text

lr_signing:

	autiasppcr w0 // 32-bit registers not allowed
	autiasppcr sp // stack pointer register not allowed

	retaasppcr w0 // 32-bit registers not allowed
	retaasppcr sp // SP not allowed
	retaasppcr xzr // zero register not allowed

	// Invalid values
	retaasppc 0xff // not multiple of 4
	retaasppc 0x3fffd // max value overflowed but might not be detected due to bit shift
	retaasppc 0x4fffc // aligned overflow: not possible to represent the value on 16 bits

	retaasppc -0x4  // unsupported negative offset
	retaasppc .+0xc // future PC target is invalid: encoded offset would be negative
	retaasppc external_symbol

	.balign 0x40000 // NOP filling

	retaasppc lr_signing // the distance between the label and the reta<k>sppc instruction
			     // is above the limit that can be encoded (=0x3fffc).
