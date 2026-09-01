# Emit add/sub rather than PC-relative relocs for symbol subtraction even sub
# symbol and pc in one frag when a R_LARCH_RELAX is set in this frag by .reloc.
  .text
1:
  add.d $t0, $t1, $t2
  add.d $t0, $t1, $t2 # R_LARCH_RELAX
  .word 0x2eef - 1b # R_LARCH_ADD32, R_LARCH_SUB32
  .dword 0x12345678 - 1b # R_LARCH_ADD64, R_LARCH_SUB64
  .4byte 0x2eef - 1b # R_LARCH_ADD32, R_LARCH_SUB32
  .8byte 0x12345678 - 1b # R_LARCH_ADD64, R_LARCH_SUB64
  .reloc 4, R_LARCH_RELAX
