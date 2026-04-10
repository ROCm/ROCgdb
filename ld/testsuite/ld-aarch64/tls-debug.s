.section .tdata,"awT",@progbits
.skip 8          // Force var to have an offset of 8
.globl var
var:
  .word 0

.section        .debug_info,"",@progbits
  .xword  %dtprel(var)
  .xword  %dtprel(var+1)
