.section .tdata,"awT",@progbits
.globl var
var:
  .word 0

.section        .debug_info,"",@progbits
  .word  %dtprel(var)
