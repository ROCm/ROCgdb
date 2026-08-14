	.text
	.seh_endprologue
	.seh_stackalloc 16
	.seh_save_reg x19, 0
	.seh_save_reg_x x19, 16
	.seh_save_regp x19, 32
	.seh_save_regp_x x19, 48
	.seh_save_lrpair x19, 64
	.seh_save_fregp d8, 80
	.seh_save_fregp_x d8, 96
	.seh_save_freg d8, 112
	.seh_save_freg_x d8, 128
	.seh_save_fplr 144
	.seh_save_fplr_x 160
	.seh_save_r19r20_x 176
	.seh_add_fp 192
	.seh_nop
	.seh_pac_sign_lr
	.seh_set_fp
	.seh_save_next
	.seh_endproc

	.seh_proc
	.seh_proc foo
	.seh_endproc

	.seh_proc foo

	.seh_stackalloc -16
	.seh_stackalloc 268435455
	.seh_stackalloc 268435456

	.seh_save_reg x18, 0
	.seh_save_reg x31, 0
	.seh_save_reg d19, 0
	.seh_save_reg x19, -8
	.seh_save_reg x19, 504
	.seh_save_reg x19, 511
	.seh_save_reg x19, 512

	.seh_save_reg_x x18, 16
	.seh_save_reg_x x31, 16
	.seh_save_reg_x d19, 16
	.seh_save_reg_x x19, -8
	.seh_save_reg_x x19, 256
	.seh_save_reg_x x19, 263
	.seh_save_reg_x x19, 264

	.seh_save_regp x18, 32
	.seh_save_regp x31, 32
	.seh_save_regp d19, 32
	.seh_save_regp x30, 32
	.seh_save_regp x19, -8
	.seh_save_regp x19, 504
	.seh_save_regp x19, 511
	.seh_save_regp x19, 512

	.seh_save_regp_x x18, 48
	.seh_save_regp_x x31, 48
	.seh_save_regp_x d19, 48
	.seh_save_regp_x x30, 48
	.seh_save_regp_x x19, -8
	.seh_save_regp_x x19, 512
	.seh_save_regp_x x19, 519
	.seh_save_regp_x x19, 520

	.seh_save_lrpair x18, 64
	.seh_save_lrpair x31, 64
	.seh_save_lrpair d19, 64
	.seh_save_lrpair x20, 64
	.seh_save_lrpair x19, -8
	.seh_save_lrpair x19, 504
	.seh_save_lrpair x19, 511
	.seh_save_lrpair x19, 512

	.seh_save_fregp d7, 80
	.seh_save_fregp d16, 80
	.seh_save_fregp x8, 80
	.seh_save_fregp d8, -8
	.seh_save_fregp d8, 504
	.seh_save_fregp d8, 511
	.seh_save_fregp d8, 512

	.seh_save_fregp_x d7, 96
	.seh_save_fregp_x d16, 96
	.seh_save_fregp_x x8, 96
	.seh_save_fregp_x d8, -8
	.seh_save_fregp_x d8, 512
	.seh_save_fregp_x d8, 519
	.seh_save_fregp_x d8, 520

	.seh_save_freg d7, 112
	.seh_save_freg d16, 112
	.seh_save_freg x8, 112
	.seh_save_freg d8, -8
	.seh_save_freg d8, 504
	.seh_save_freg d8, 511
	.seh_save_freg d8, 512

	.seh_save_freg_x d7, 128
	.seh_save_freg_x d16, 128
	.seh_save_freg_x x8, 128
	.seh_save_freg_x d8, -8
	.seh_save_freg_x d8, 256
	.seh_save_freg_x d8, 263
	.seh_save_freg_x d8, 264

	.seh_save_fplr -8
	.seh_save_fplr 504
	.seh_save_fplr 511
	.seh_save_fplr 512

	.seh_save_fplr_x -8
	.seh_save_fplr_x 512
	.seh_save_fplr_x 519
	.seh_save_fplr_x 520

	.seh_save_r19r20_x -8
	.seh_save_r19r20_x 248
	.seh_save_r19r20_x 255
	.seh_save_r19r20_x 256

	.seh_add_fp -8
	.seh_add_fp 2040
	.seh_add_fp 2047
	.seh_add_fp 2048

	.seh_endprologue
