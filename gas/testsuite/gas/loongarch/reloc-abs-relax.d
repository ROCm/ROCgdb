#source: reloc-abs-relax.s
#as: -mthin-add-sub -mrelax
#objdump: -dr

#...
.*8: R_LARCH_ADD16	\*ABS\*\+0x2eef
.*8: R_LARCH_SUB16	.L1\^B1
.*a: R_LARCH_ADD32	\*ABS\*\+0x2eef
.*a: R_LARCH_SUB32	.L1\^B1
.*e: R_LARCH_ADD64	\*ABS\*\+0x12345678
.*e: R_LARCH_SUB64	.L1\^B1
.*16: R_LARCH_ADD16	\*ABS\*\+0x2eef
.*16: R_LARCH_SUB16	.L1\^B1
.*18: R_LARCH_ADD32	\*ABS\*\+0x2eef
.*18: R_LARCH_SUB32	.L1\^B1
.*1c: R_LARCH_ADD8	\*ABS\*\+0x2e
.*1c: R_LARCH_SUB8	.L1\^B1
.*1d: R_LARCH_ADD16	\*ABS\*\+0x2eef
.*1d: R_LARCH_SUB16	.L1\^B1
.*1f: R_LARCH_ADD32	\*ABS\*\+0x2eef
.*1f: R_LARCH_SUB32	.L1\^B1
.*23: R_LARCH_ADD64	\*ABS\*\+0x12345678
.*23: R_LARCH_SUB64	.L1\^B1
