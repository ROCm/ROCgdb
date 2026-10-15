# .reloc R_LARCH_RELAX disables align optimization even before a relaxable
# instruction.
  .text
.L1:
  add.d $t0, $t1, $t2
  .align 4 # R_LARCH_ALIGN
.L2:
  add.d $t0, $t1, $t2 # R_LARCH_RELAX
  .reloc .L2, R_LARCH_RELAX
  .4byte .L2 - .L1 # R_LARCH_ADD32, R_LARCH_SUB32
