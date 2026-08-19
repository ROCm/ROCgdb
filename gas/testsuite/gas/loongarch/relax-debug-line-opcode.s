  # Test DWARF line encoding around linker-relaxable instructions.
  .file 0 "test"
  .text
  .loc 0 10 0
  nop		# special opcode
  .loc 0 11 0
  call36 func	# -mrelax: DW_LNS_fixed_advance_pc; -mno-relax: special opcode.
  .loc 0 12 0
  nop		# special opcode
  .loc 0 13 0
  call36 func	# -mrelax: DW_LNS_fixed_advance_pc; -mno-relax: special opcode.
  .loc 0 14 0
  nop		# Advance PC

.section .debug_line, "", @progbits
