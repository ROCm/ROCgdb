#source: reloc-abs.s
#as: -mrelax -mthin-add-sub
#objdump: -dr

#...
.*R_LARCH_ADD16	\*ABS\*\+0x2eef
.*R_LARCH_SUB16	.L1\^B1
.*R_LARCH_32_PCREL	\*ABS\*\+0x2ef5
.*R_LARCH_64_PCREL	\*ABS\*\+0x12345682
.*R_LARCH_ADD16	\*ABS\*\+0x2eef
.*R_LARCH_SUB16	.L1\^B1
.*R_LARCH_32_PCREL	\*ABS\*\+0x2f03
.*R_LARCH_ADD8	\*ABS\*\+0x2e
.*R_LARCH_SUB8	.L1\^B1
.*R_LARCH_ADD16	\*ABS\*\+0x2eef
.*R_LARCH_SUB16	.L1\^B1
.*R_LARCH_32_PCREL	\*ABS\*\+0x2f0a
.*R_LARCH_64_PCREL	\*ABS\*\+0x12345697
