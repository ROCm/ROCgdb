# Use DW_LNS_fixed_advance_pc even from symbol and to symbol in one frag when
# a R_LARCH_RELAX is set in this frag by .reloc.
  .file 0 "t.c"
  .text
  .loc 0 10 0
  nop
  .loc 0 11 0
  add.d $t0, $t1, $t2
  .loc 0 12 0
  nop
.L1:
  nop
  .reloc .L1, R_LARCH_RELAX

.section .debug_line, "", @progbits
