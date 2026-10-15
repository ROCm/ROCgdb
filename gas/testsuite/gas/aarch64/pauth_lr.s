	/* FEAT_PAuth_LR Pointer Authentication instructions.  */
	.text

	// only NOP-space instruction to enable ComputePAC2() with PAuth instructions.
	pacm

	// Signing of LR using SP and PC as diversifiers.
	paciasppc
	pacibsppc

	pacia171615
	pacib171615

	// When used along feature FEAT_BTI
	pacnbiasppc
	pacnbibsppc

	// Authenticating the PAC in the address in register LR
	autia171615
	autib171615

	// Variant with value of PC passed by register
	autiasppcr x0
	autiasppcr xzr
	autiasppcr x30
	autibsppcr x0

	// Authentication + return
	retaasppcr x0
	retaasppcr x30
	retabsppcr x0

lr_signing:
	autiasppc lr_signing
	autibsppc lr_signing
	retaasppc lr_signing
	retabsppc lr_signing

	retaasppc 0x0 // min: reference to itself
	retaasppc .+0x0 // min: current PC + 0
	retaasppc 0x4 // first valid value after min: point to previous instruction
	retaasppc 0x20
	retaasppc .-0x58 // point to pacm

	.balign 0x40000 // NOP filling

	retaasppc 0x1234c // a random value in the valid range
	retaasppc 0x3fffc // max
