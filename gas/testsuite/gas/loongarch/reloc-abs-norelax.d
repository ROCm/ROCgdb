#source: reloc-abs-relax.s
#as: -mthin-add-sub -mno-relax
#objdump: -dr

#...
.*8: R_LARCH_ADD16	\*ABS\*\+0x2eef
.*8: R_LARCH_SUB16	.L1\^B1
.*a: R_LARCH_32_PCREL	\*ABS\*\+0x2eef
.*e: R_LARCH_64_PCREL	\*ABS\*\+0x12345678
.*16: R_LARCH_ADD16	\*ABS\*\+0x2eef
.*16: R_LARCH_SUB16	.L1\^B1
.*18: R_LARCH_32_PCREL	\*ABS\*\+0x2eef
.*1c: R_LARCH_ADD8	\*ABS\*\+0x2e
.*1c: R_LARCH_SUB8	.L1\^B1
.*1d: R_LARCH_ADD16	\*ABS\*\+0x2eef
.*1d: R_LARCH_SUB16	.L1\^B1
.*1f: R_LARCH_32_PCREL	\*ABS\*\+0x2eef
.*23: R_LARCH_64_PCREL	\*ABS\*\+0x12345678
