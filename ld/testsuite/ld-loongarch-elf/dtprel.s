.section .tdata,"awT",@progbits
.skip 8
var:
  .word 0

.section        .debug_info,"",@progbits
  .dtprelword var
  .dtprelword var+4
  .dtpreldword var
  .dtpreldword var+8
