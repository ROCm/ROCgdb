# .reloc R_LARCH_RELAX disables align optimization even before a relaxable
# instruction.
  .text
.L1:
  .align 3
  add.d $t0, $t1, $t2
  .align 4
  add.d $t0, $t1, $t2
.L2:
  .4byte .L2 - .L1
