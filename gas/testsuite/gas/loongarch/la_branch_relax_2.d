#as: -mrelax
#objdump: -dr

#...
.*beq.*
.*R_LARCH_B16.*
.*bne.*
.*b .*
.*R_LARCH_B26.*
.*bne.*
.*b .*
.*R_LARCH_B26.*
.*beq.*
.*R_LARCH_B16.*
#...
.*beqz.*
.*R_LARCH_B21.*
.*bnez.*
.*b .*
.*R_LARCH_B26.*
.*bnez.*
.*b .*
.*R_LARCH_B26.*
.*beqz.*
.*R_LARCH_B21.*
#...
.*beq.*t0.*t1, 0.*
.*beqz.*t0, 0.*
