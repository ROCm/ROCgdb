# Emit add/sub relocs for symbol subtraction even two symbols in one frag
# when a R_LARCH_RELAX is set in this frag by .reloc.
  .text
.L1:
  nop
.L2:
  nop # R_LARCH_RELAX
.L3:
  .4byte .L3 - .L1 # R_LARCH_ADD32,R_LARCH_SUB32
  .8byte .L3 - .L1 # R_LARCH_ADD64,R_LARCH_SUB64
  .reloc .L2, R_LARCH_RELAX
