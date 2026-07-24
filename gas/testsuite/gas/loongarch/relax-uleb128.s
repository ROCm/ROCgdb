.text
.L1:
  nop
  call .L1
  nop
.L2:
  nop
  .p2align 3
  nop
.L3:
  nop
.L4:
  nop

.data
  .uleb128 .L2-.L1 # Emit add/sub uleb128 relocs.
  .uleb128 .L3-.L2 # Emit add/sub uleb128 relocs.
  .uleb128 .L4-.L3 # Not emit add/sub uleb128 relocs.
