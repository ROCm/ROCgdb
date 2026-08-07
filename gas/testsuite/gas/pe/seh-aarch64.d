#objdump: -s -j .xdata
#name: PEP aarch64 SEH

.*:     file format pe-aarch64-little

Contents of section .xdata:

# .xdata SEH record
# 0x98500000  code-words: 19
#             epilogue-count: 1
#             single-epilogue-in-header: 0
#             exception-data: 1
#             version: 0
#             function-size: 0
# 0x00000000  start-index: 0
#             epilog-start-offset: 0
# 0xe6        .seh_save_next
# 0xe1        .seh_set_fp
# 0xfc        .seh_pac_sign_lr
# 0xe3        .seh_nop
# 0xe218      .seh_add_fp 192
# 0x36        .seh_save_r19r20_x 176
# 0x93        .seh_save_fplr_x 160
# 0x52        .seh_save_fplr 144
# 0xdeef      .seh_save_freg_x d15, 128
# 0xde2f      .seh_save_freg_x d9, 128
# 0xde0f      .seh_save_freg_x d8, 128
# 0xddce      .seh_save_freg d15, 112
# 0xdc4e      .seh_save_freg d9, 112
# 0xdc0e      .seh_save_freg d8, 112
# 0xdbcb      .seh_save_fregp_x d15, 96
# 0xda4b      .seh_save_fregp_x d9, 96
# 0xda0b      .seh_save_fregp_x d8, 96
# 0xd9ca      .seh_save_fregp d15, 80
# 0xd84a      .seh_save_fregp d9, 80
# 0xd80a      .seh_save_fregp d8, 80
# 0xd748      .seh_save_lrpair x29, 64
# 0xd648      .seh_save_lrpair x21, 64
# 0xd608      .seh_save_lrpair x19, 64
# 0xce85      .seh_save_regp_x x29, 48
# 0xcc45      .seh_save_regp_x x20, 48
# 0xcc05      .seh_save_regp_x x19, 48
# 0xca84      .seh_save_regp x29, 32
# 0xc844      .seh_save_regp x20, 32
# 0xc804      .seh_save_regp x19, 32
# 0xd561      .seh_save_reg_x x30, 16
# 0xd421      .seh_save_reg_x x20, 16
# 0xd401      .seh_save_reg_x x19, 16
# 0xd2c0      .seh_save_reg x30, 0
# 0xd040      .seh_save_reg x20, 0
# 0xd000      .seh_save_reg x19, 0
# 0xe0ffffff  .seh_stackalloc 268435440
# 0xe0000800  .seh_stackalloc 32768
# 0xc020      .seh_stackalloc 512
# 0x01        .seh_stackalloc 16
# 0xe4        nop
# 0xe4        nop
# 0x00000000  seh-handler
# 0x00000000  fragment-offset
# 0x00000060  seh-handle-data-address
# 0x00000001  long 1 (seh_handlerdata)

 0000 00005098 00000000 e6e1fce3 e2183693  .*
 0010 52deefde 2fde0fdd cedc4edc 0edbcbda  .*
 0020 4bda0bd9 cad84ad8 0ad748d6 48d608ce  .*
 0030 85cc45cc 05ca84c8 44c804d5 61d421d4  .*
 0040 01d2c0d0 40d000e0 ffffffe0 000800c0  .*
 0050 2001e4e4 00000000 00000000 60000000  .*
 0060 01000000 .*