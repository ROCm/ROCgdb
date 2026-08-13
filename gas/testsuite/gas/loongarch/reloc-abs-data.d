#source: reloc-abs-data.s
#as: -mthin-add-sub
#objdump: -Dr

#...
.*R_LARCH_ADD16	\*ABS\*\+0x2eef
.*R_LARCH_SUB16	.L1\^B1
.*R_LARCH_32_PCREL	\*ABS\*\+0x2eef
.*R_LARCH_64_PCREL	\*ABS\*\+0x12345678
.*R_LARCH_ADD16	\*ABS\*\+0x2eef
.*R_LARCH_SUB16	.L1\^B1
.*R_LARCH_32_PCREL	\*ABS\*\+0x2eef
.*R_LARCH_ADD8	\*ABS\*\+0x2e
.*R_LARCH_SUB8	.L1\^B1
.*R_LARCH_ADD16	\*ABS\*\+0x2eef
.*R_LARCH_SUB16	.L1\^B1
.*R_LARCH_32_PCREL	\*ABS\*\+0x2eef
.*R_LARCH_64_PCREL	\*ABS\*\+0x12345678
