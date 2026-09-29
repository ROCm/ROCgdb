# Test that a local branch can still be relaxed at link time even when
# relaxation is disabled for the branch instruction.

.text
.L1:
  .option relax
  call36 x
  .align 5
  add.d $t0, $t0, $t0
  .option norelax
  b .L1
