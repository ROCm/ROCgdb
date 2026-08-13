#source: reloc-add-sub.s
#as: -mrelax
#objdump: -dr

#...
			4: R_LARCH_ADD32	\*ABS\*\+0xa
			4: R_LARCH_SUB32	x
			8: R_LARCH_32	x-0xa
			c: R_LARCH_ADD32	x
			c: R_LARCH_SUB32	y
			10: R_LARCH_32	x
  14:	00000005 	.word		0x00000005
			15: R_LARCH_ADD8	\*ABS\*\+0xa
			15: R_LARCH_SUB8	x
			16: R_LARCH_ADD8	x-0xa
			17: R_LARCH_ADD8	x
			17: R_LARCH_SUB8	y
  18:	00000000 	.word		0x00000000
			18: R_LARCH_ADD8	x
			19: R_LARCH_ADD8	x
			1a: R_LARCH_ADD8	x
			1b: R_LARCH_ADD8	x
