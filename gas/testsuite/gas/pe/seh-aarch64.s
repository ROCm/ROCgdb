	.text
	.seh_proc	foo
	.seh_stackalloc 16
	.seh_stackalloc 512
	.seh_stackalloc 32768
	.seh_stackalloc 268435440
	.seh_save_reg x19, 0
	.seh_save_reg x20, 0
	.seh_save_reg x30, 0
	.seh_save_reg_x x19, 16
	.seh_save_reg_x x20, 16
	.seh_save_reg_x x30, 16
	.seh_save_regp x19, 32
	.seh_save_regp x20, 32
	.seh_save_regp x29, 32
	.seh_save_regp_x x19, 48
	.seh_save_regp_x x20, 48
	.seh_save_regp_x x29, 48
	.seh_save_lrpair x19, 64
	.seh_save_lrpair x21, 64
	.seh_save_lrpair x29, 64
	.seh_save_fregp d8, 80
	.seh_save_fregp d9, 80
	.seh_save_fregp d15, 80
	.seh_save_fregp_x d8, 96
	.seh_save_fregp_x d9, 96
	.seh_save_fregp_x d15, 96
	.seh_save_freg d8, 112
	.seh_save_freg d9, 112
	.seh_save_freg d15, 112
	.seh_save_freg_x d8, 128
	.seh_save_freg_x d9, 128
	.seh_save_freg_x d15, 128
	.seh_save_fplr 144
	.seh_save_fplr_x 160
	.seh_save_r19r20_x 176
	.seh_add_fp 192
	.seh_nop
	.seh_pac_sign_lr
	.seh_set_fp
	.seh_save_next
	.seh_endprologue
	.seh_handler _ZN9exception6handleEPvS0_S0_S0_, @except
	.seh_handlerdata
	.long 1
	.seh_code
	.seh_startepilogue
	.seh_endepilogue
	.seh_endproc
