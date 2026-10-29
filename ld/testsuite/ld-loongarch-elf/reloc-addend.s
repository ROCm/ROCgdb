  .text
  .globl _start
_start:
  nop
  .reloc _start, R_LARCH_SOP_SR, _start - 0x8000000000000000
