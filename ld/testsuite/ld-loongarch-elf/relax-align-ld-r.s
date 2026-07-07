# Add an align section before text section when ld -r.
# There will be two R_LARCH_ALIGN relocations after ld -r.
.text
  addi.d $a0, $a0, 1
  call36 func
  .align 3
func:
  addi.d $a0, $a0, 1

# Only add a align section for code section.
.data
  .4byte 0x1234
