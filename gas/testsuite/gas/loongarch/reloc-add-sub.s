  .text
x:
  .long 10 - 5	# no relocation
  .long 10 - x	# R_LARCH_ADD32/SUB32
  .long x - 10	# R_LARCH_32
  .long x - y	# R_LARCH_ADD32/SUB32
  .long x	# R_LARCH_32

  .byte 10 - 5	# no relocation
  .byte 10 - x	# R_LARCH_ADD8/SUB8
  .byte x - 10	# R_LARCH_ADD8
  .byte x - y   # R_LARCH_ADD8/SUB8
  .byte x	# R_LARCH_ADD8
  .byte x
  .byte x
  .byte x
