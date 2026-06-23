# Emit align relocation only if the current section
# already has relaxed instrctions.
.text
  addi.d $t0, $t0, 1
  .align 3 # Do not emit an align relocation
  call .text
  .align 3 # Emit an align relocation
  addi.d $t0, $t0, 1

.section ".text.a", "ax"
  addi.d $t0, $t0, 1
  .align 3 # Do not emit an align relocation
  call .text
  .align 3 # Emit an align relocation
  addi.d $t0, $t0, 1

