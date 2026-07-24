# Immediate boundary value tests

.text
.L1:
  .fill 0x7ffe, 4, 0
  call .L1
  beq $r12, $r13, .L1 # min imm -0x20000
  beq $r12, $r13, .L1 # out of range, chang to bne+b
  beq $r12, $r13, .L2 # out of range, change to bne+b
  beq $r12, $r13, .L2 # max imm 0x1fffc
  call .L1
  .fill 0x7ffc, 4, 0
.L2:
  .fill 0xffffe, 4, 0
  call .L1
  beqz $r12, .L2 # min imm -0x400000
  beqz $r12, .L2 # out of range, change to bnez+b
  beqz $r12, .L3 # out of range, change to bnez+b
  beqz $r12, .L3 # max imm 0x3ffffc
  call .L1
  .fill 0xffffc, 4, 0
.L3:

# 0 imm: branch target is current pc
.L4:
  beq $r12, $r13, .L4
.L5:
  beqz $r12, .L5
