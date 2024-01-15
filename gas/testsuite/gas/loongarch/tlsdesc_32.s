.L1:
	# R_LARCH_TLS_DESC_PC_HI20 var
	pcalau12i       $a0,%desc_pc_hi20(var)
	# R_LARCH_TLS_DESC_PC_LO12 var
	addi.w  $a0,$a0,%desc_pc_lo12(var)
	# R_LARCH_TLS_DESC_LD var
        ld.w    $ra,$a0,%desc_ld(var)
	# R_LARCH_TLS_DESC_CALL var
	jirl    $ra,$ra,%desc_call(var)

	# test macro, pcalau12i + addi.w => pcaddi
	la.tls.desc	$a0,var
